"""
Temple Run AI Configuration Loader
This file contains only logic to load and process configuration.
Edit config.yaml for all configuration parameters.
"""

import yaml
import json
import os
from pathlib import Path

# Load configuration from YAML file
config_path = Path(__file__).parent / "config.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Apply GPU preset
gpu_preset = config['gpu_preset']
if gpu_preset not in config['gpu_presets']:
    raise ValueError(f"Invalid GPU preset: {gpu_preset}. Available: {list(config['gpu_presets'].keys())}")

preset = config['gpu_presets'][gpu_preset]

# GPU/Hardware Settings
RESOLUTION = preset['resolution']
CLASSIFIER_BATCH_SIZE = preset['classifier_batch_size']
PPO_BATCH_SIZE = preset['ppo_batch_size']
FRAME_SKIP = preset['frame_skip']
USE_FP16 = preset['use_fp16']

# Training Configuration
TOTAL_TIMESTEPS = config['training']['total_timesteps']
CHECKPOINT_INTERVAL = config['training']['checkpoint_interval']
EVAL_FREQUENCY = config['training']['eval_frequency']

# Hyperparameter Loading Logic
USE_OPTIMIZED_PARAMS = config['use_optimized_params']

if USE_OPTIMIZED_PARAMS:
    # Try to load optimized parameters from best_params.json
    best_params_path = "best_params.json"
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
            
        # Load optimized PPO hyperparameters
        LEARNING_RATE = best_params.get('learning_rate', config['ppo']['learning_rate'])
        N_STEPS = best_params.get('n_steps', config['ppo']['n_steps'])
        GAMMA = best_params.get('gamma', config['ppo']['gamma'])
        ENT_COEF = best_params.get('ent_coef', config['ppo']['ent_coef'])
        CLIP_RANGE = best_params.get('clip_range', config['ppo']['clip_range'])
        VF_COEF = best_params.get('vf_coef', config['ppo']['vf_coef'])
        
        print(f"✓ Using optimized hyperparameters from {best_params_path}")
    else:
        # Fall back to manual values from YAML
        LEARNING_RATE = config['ppo']['learning_rate']
        N_STEPS = config['ppo']['n_steps']
        GAMMA = config['ppo']['gamma']
        ENT_COEF = config['ppo']['ent_coef']
        CLIP_RANGE = config['ppo']['clip_range']
        VF_COEF = config['ppo']['vf_coef']
        
        print("! No best_params.json found - using manual hyperparameters from config.yaml")
else:
    # Use manual hyperparameters from YAML
    LEARNING_RATE = config['ppo']['learning_rate']
    N_STEPS = config['ppo']['n_steps']
    GAMMA = config['ppo']['gamma']
    ENT_COEF = config['ppo']['ent_coef']
    CLIP_RANGE = config['ppo']['clip_range']
    VF_COEF = config['ppo']['vf_coef']
    
    print("✓ Using manual hyperparameters from config.yaml")

# Additional PPO hyperparameters (not typically optimized)
GAE_LAMBDA = config['ppo']['gae_lambda']
N_EPOCHS = config['ppo']['n_epochs']
MAX_GRAD_NORM = config['ppo']['max_grad_norm']

# Classifier hyperparameters
CLASSIFIER_LEARNING_RATE = config['classifier']['learning_rate']
CLASSIFIER_EPOCHS = config['classifier']['epochs']
CLASSIFIER_WARMUP_STEPS = config['classifier']['warmup_steps']
CLASSIFIER_WEIGHT_DECAY = config['classifier']['weight_decay']

# Environment hyperparameters
MAX_EPISODE_STEPS = config['environment']['max_episode_steps']
REWARD_SCALE = config['environment']['reward_scale']
ACTION_REPEAT = config['environment']['action_repeat']

# Device configuration
DEVICE_ID = config['device']['device_id']
WINDOW_TITLE = config['device']['window_title']

# Data paths
DATASET_PATH = config['paths']['dataset_path']

# Derived settings
OBSERVATION_SHAPE = (3, RESOLUTION, RESOLUTION)
STATE_SPACE_DIM = RESOLUTION * RESOLUTION * 3

# Configuration summary
print(f"""
Temple Run AI Configuration Loaded:
===================================
GPU Preset: {gpu_preset}
Resolution: {RESOLUTION}x{RESOLUTION}
State Space: {STATE_SPACE_DIM:,} dimensions
Classifier Batch: {CLASSIFIER_BATCH_SIZE}
PPO Batch: {PPO_BATCH_SIZE}
Frame Skip: {FRAME_SKIP}
Mixed Precision: {USE_FP16}
Total Steps: {TOTAL_TIMESTEPS:,}
Optimized Params: {USE_OPTIMIZED_PARAMS}
===================================
""")