import sys
import os
import shutil
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
)
from stable_baselines3 import PPO
import optuna
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from temple_run.env import TempleRunEnv
from temple_run.agent import TempleRunAgent
from training.dataset_loader import load_templerun_dataset
from training_time_tracker import TrainingTimeTracker, TrainingTimeCallback
import glob
import json
import datetime
import time

# Import centralized config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temple_run.config import (
    CLASSIFIER_BATCH_SIZE, PPO_BATCH_SIZE, TOTAL_TIMESTEPS, 
    CHECKPOINT_INTERVAL, EVAL_FREQUENCY, LEARNING_RATE,
    N_STEPS, GAMMA, ENT_COEF, CLIP_RANGE, VF_COEF, USE_FP16
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(labels, predictions, target_names=['alive', 'dead']))
    print(confusion_matrix(labels, predictions))
    return {'accuracy': accuracy}

class ProcessorSavingCallback(TrainerCallback):
    def __init__(self, processor, verbose=0):
        super().__init__()
        self.processor = processor
        self.verbose = verbose

    def on_save(self, args, state, control, **kwargs):
        output_dir = args.output_dir
        if state.best_model_checkpoint:
            self.processor.save_pretrained(state.best_model_checkpoint)
            if self.verbose:
                print(f"Saved processor to {state.best_model_checkpoint}")
        
        if state.is_world_process_zero:
            checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
            if checkpoint_dirs:
                checkpoint_nums = [int(d.split('-')[1]) for d in checkpoint_dirs]
                latest_checkpoint_num = max(checkpoint_nums)
                latest_checkpoint_dir = os.path.join(output_dir, f"checkpoint-{latest_checkpoint_num}")
                self.processor.save_pretrained(latest_checkpoint_dir)
                if self.verbose:
                    print(f"Saved processor to {latest_checkpoint_dir}")

def load_classifier_and_processor(classifier_dir, device):
    last_checkpoint = None
    model = None
    loaded_processor = None
    
    if os.path.exists(classifier_dir):
        print("Checking for existing classifier...")
        try:
            if os.path.exists(os.path.join(classifier_dir, "config.json")) and \
               os.path.exists(os.path.join(classifier_dir, "preprocessor_config.json")):
                model = AutoModelForImageClassification.from_pretrained(
                    classifier_dir,
                    num_labels=2,
                    id2label={0: 'alive', 1: 'dead'},
                    label2id={'alive': 0, 'dead': 1}
                ).to(device)
                loaded_processor = AutoImageProcessor.from_pretrained(classifier_dir)
                print("Loaded classifier and processor from main directory")
                return model, loaded_processor, None
        except Exception as e:
            print(f"Could not load from main directory: {e}")
        
        checkpoint_dirs = [d for d in os.listdir(classifier_dir) if d.startswith('checkpoint-')]
        if checkpoint_dirs:
            checkpoint_nums = [int(d.split('-')[1]) for d in checkpoint_dirs]
            latest_checkpoint_num = max(checkpoint_nums)
            last_checkpoint = os.path.join(classifier_dir, f"checkpoint-{latest_checkpoint_num}")
            print(f"Found checkpoint: {last_checkpoint}")
            
            try:
                model = AutoModelForImageClassification.from_pretrained(
                    last_checkpoint,
                    num_labels=2,
                    id2label={0: 'alive', 1: 'dead'},
                    label2id={'alive': 0, 'dead': 1}
                ).to(device)
                
                if os.path.exists(os.path.join(last_checkpoint, "preprocessor_config.json")):
                    loaded_processor = AutoImageProcessor.from_pretrained(last_checkpoint)
                    print("Loaded both model and processor from checkpoint")
                else:
                    print("Checkpoint missing processor config, will use original processor")
                
                return model, loaded_processor, last_checkpoint
                
            except Exception as e:
                print(f"Failed to load from checkpoint {last_checkpoint}: {e}")
    
    return None, None, None

def objective(trial, model, processor, device):
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    gamma = trial.suggest_float("gamma", 0.95, 0.999, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    vf_coef = trial.suggest_float("vf_coef", 0.25, 1.0)
    
    env = None
    try:
        agent = TempleRunAgent(create_new=True)
        env = TempleRunEnv(model, processor, agent=agent)
        agent.initialize_model(env)
        check_env(env)
        env = Monitor(env)
        
        ppo_model = PPO(
            "CnnPolicy", env,
            verbose=0,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range=clip_range,
            vf_coef=vf_coef,
        )
        
        ppo_model.learn(total_timesteps=5000)
        mean_reward, _ = evaluate_policy(ppo_model, env, n_eval_episodes=5)
        
        return mean_reward
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('-inf')
    finally:
        if env:
            env.close()

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0, log_path="training_output/episode_log.json", total_timesteps=50000):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.log_path = log_path
        self.total_reward = 0
        self.episode_count = self._load_episode_count()
        self.total_timesteps = total_timesteps
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _load_episode_count(self):
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    data = json.load(f)
                    count = data.get("episode_count", 0)
                    print(f"[INFO] Resuming episode count from {count}")
                    return count
            except (json.JSONDecodeError, IOError):
                print("[WARNING] Could not read episode log file. Starting from episode 0.")
                return 0
        return 0

    def _save_episode_count(self):
        try:
            with open(self.log_path, 'w') as f:
                json.dump({"episode_count": self.episode_count}, f)
        except IOError as e:
            print(f"[ERROR] Could not save episode count: {e}")

    def _on_training_start(self) -> None:
        print(f"[TRAINING START] Total timesteps to run: {self.total_timesteps}")

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.total_reward += reward
        print(f"[PROGRESS] Steps: {self.num_timesteps}/{self.total_timesteps}")
        
        if self.locals['dones'][0]:
            self.episode_count += 1
            print(f"[EPISODE {self.episode_count}] Finished at step {self.num_timesteps}. Episode Reward: {self.total_reward:.2f}")
            self.total_reward = 0
            self._save_episode_count()
        return True

    def _on_training_end(self) -> None:
        print(f"[TRAINING END] Total Episodes Completed: {self.episode_count}")
        print(f"[TRAINING END] Total Steps: {self.num_timesteps}/{self.total_timesteps}")
        self._save_episode_count()

class CurriculumCallback(BaseCallback):
    def __init__(self, check_freq, reward_threshold, new_max_steps, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.reward_threshold = reward_threshold
        self.new_max_steps = new_max_steps
        self.updated = False

    def _on_step(self) -> bool:
        if self.updated:
            return True

        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                if mean_reward > self.reward_threshold:
                    print(f"[Curriculum] Increasing max_episode_steps to {self.new_max_steps}")
                    if hasattr(self.training_env, 'env_method'):
                        self.training_env.env_method("set_max_episode_steps", self.new_max_steps)
                    self.updated = True
        return True

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.save_freq == 0:
            checkpoint_file = os.path.join(self.save_path, f"ppo_step_{self.num_timesteps}.zip")
            self.model.save(checkpoint_file)
            print(f"[Checkpoint] Saved to {checkpoint_file}")
        return True

def get_latest_checkpoint(checkpoint_dir, env):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "ppo_step_*.zip"))
    compatible_checkpoints = []
    for checkpoint in checkpoint_files:
        try:
            PPO.load(checkpoint, env=env)
            compatible_checkpoints.append(checkpoint)
        except Exception:
            pass
    
    if compatible_checkpoints:
        return max(compatible_checkpoints, key=os.path.getctime)
    return None

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading and preprocessing dataset...")
    from config import DATASET_PATH
    ds = load_templerun_dataset(DATASET_PATH)
    
    if 'test' not in ds:
        split = ds['train'].train_test_split(test_size=0.2, seed=42)
        train_dataset, eval_dataset = split['train'], split['test']
    else:
        train_dataset, eval_dataset = ds['train'], ds['test']

    processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', use_fast=True)

    def preprocess(examples):
        images = [img.convert('RGB') for img in examples['image']]
        inputs = processor(images, return_tensors='pt')
        inputs['labels'] = torch.tensor(examples['label'], dtype=torch.long)
        return inputs

    train_dataset = train_dataset.map(preprocess, batched=True, batch_size=8, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess, batched=True, batch_size=8, remove_columns=eval_dataset.column_names)

    classifier_dir = './templerun-classifier'
    model, loaded_processor, last_checkpoint = load_classifier_and_processor(classifier_dir, device)

    if model is None:
        print("Training new classifier...")
        model = AutoModelForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=2, id2label={0: 'alive', 1: 'dead'}, label2id={'alive': 0, 'dead': 1}
        ).to(device)
        
        args = TrainingArguments(
            output_dir=classifier_dir,
            per_device_train_batch_size=CLASSIFIER_BATCH_SIZE, 
            per_device_eval_batch_size=CLASSIFIER_BATCH_SIZE,
            evaluation_strategy='steps', eval_steps=50,
            save_strategy='steps', save_steps=500, save_total_limit=3,
            num_train_epochs=10, learning_rate=2e-5, warmup_steps=100,
            logging_steps=25, load_best_model_at_end=True,
            metric_for_best_model='eval_accuracy', greater_is_better=True,
            fp16=USE_FP16 and torch.cuda.is_available(), 
            dataloader_pin_memory=False,
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=model, args=args,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(3, 0.01), ProcessorSavingCallback(processor, verbose=1)]
        )
        
        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(classifier_dir)
        processor.save_pretrained(classifier_dir)
        print("Classifier training completed.")
    else:
        if loaded_processor:
            processor = loaded_processor
        print("Classifier loaded successfully.")

    best_params_path = "best_params.json"
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        print("Loaded best hyperparameters:", best_params)
    else:
        print("Starting hyperparameter optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model, processor, device), n_trials=10)
        best_params = study.best_params
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f)
        print("Best hyperparameters found and saved:", best_params)

    print("\n" + "="*50)
    print("Starting PPO RL Training")
    print("="*50)

    output_dir = "training_output"
    checkpoint_dir = os.path.join(output_dir, "ppo_checkpoints")
    best_model_dir = os.path.join(output_dir, "best_model")
    agent_path = os.path.join(output_dir, "templerun_agent.zip")
    eval_log_path = os.path.join(output_dir, "eval_logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_log_path, exist_ok=True)

    train_env = None
    eval_env = None
    
    try:
        agent = TempleRunAgent()
        train_env = Monitor(TempleRunEnv(model, processor, agent=agent))
        eval_env = Monitor(TempleRunEnv(model, processor, agent=agent))

        latest_ppo_checkpoint = get_latest_checkpoint(checkpoint_dir, train_env)

        if latest_ppo_checkpoint:
            print(f"Loading latest compatible checkpoint: {latest_ppo_checkpoint}")
            ppo_model = PPO.load(latest_ppo_checkpoint, env=train_env)
            print(f"Resuming training from timestep: {ppo_model.num_timesteps}")
        else:
            print("Creating new PPO model with best hyperparameters...")
            ppo_model = PPO("CnnPolicy", train_env, verbose=1, tensorboard_log="./templerun_tensorboard/", **best_params)

        time_tracker = TrainingTimeTracker(log_file="training_time_log.json")
        
        callbacks = [
            EvalCallback(eval_env, best_model_save_path=best_model_dir, log_path=eval_log_path, eval_freq=2000, deterministic=True, render=False),
            RewardLoggerCallback(total_timesteps=TOTAL_TIMESTEPS),
            CheckpointCallback(save_freq=CHECKPOINT_INTERVAL, save_path=checkpoint_dir),
            CurriculumCallback(check_freq=10000, reward_threshold=50, new_max_steps=30),
            TrainingTimeCallback(time_tracker, log_freq=1000, total_timesteps=TOTAL_TIMESTEPS, verbose=1)
        ]

        remaining_timesteps = TOTAL_TIMESTEPS - ppo_model.num_timesteps
        
        if remaining_timesteps > 0:
            print(f"Training for an additional {remaining_timesteps} timesteps...")
            ppo_model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=not latest_ppo_checkpoint
            )
        else:
            print("Training already completed.")
        
        ppo_model.save(agent_path)
        print(f"Final model saved to: {agent_path}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        if 'ppo_model' in locals():
            emergency_save_path = os.path.join(output_dir, "emergency_save.zip")
            ppo_model.save(emergency_save_path)
            print(f"Emergency model saved to {emergency_save_path}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model as a checkpoint...")
        if 'ppo_model' in locals():
            checkpoint_file = os.path.join(checkpoint_dir, f"ppo_step_{ppo_model.num_timesteps}.zip")
            ppo_model.save(checkpoint_file)
            print(f"Checkpoint saved to {checkpoint_file}")
    finally:
        print("Closing environments...")
        if train_env:
            train_env.close()
        if eval_env:
            eval_env.close()
        print("Training script finished.")

if __name__ == '__main__':
    main()
