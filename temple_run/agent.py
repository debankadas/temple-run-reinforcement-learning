from stable_baselines3 import PPO
import os
import gymnasium as gym
import numpy as np

class TempleRunAgent:
    def __init__(self, model_path="training_output/best_model/best_model.zip", create_new=False):
        """
        Initialize the Temple Run agent.
        
        Args:
            model_path: Path to the pre-trained model
            create_new: If True, create a new model instead of loading an existing one
        """
        self.model_path = model_path
        
        if create_new:
            print(f"Creating new PPO agent (no model will be loaded)")
            # We'll initialize the model later when we have the environment
            self.model = None
        else:
            # Try to load the model, but handle the case where it doesn't exist
            try:
                if not os.path.exists(model_path):
                    print(f"Warning: Model not found at {model_path}")
                    print("Using random actions instead")
                    self.model = None
                else:
                    print(f"Loading model from {model_path}")
                    self.model = PPO.load(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using random actions instead")
                self.model = None

    def initialize_model(self, env):
        """Initialize a new model with the given environment"""
        if self.model is None:
            print("Initializing new PPO model")
            self.model = PPO("CnnPolicy", env, verbose=1)
            return True
        return False

    def predict(self, obs, deterministic=True):
        """Get action prediction from the model or random action if no model"""
        if self.model is not None:
            action, _states = self.model.predict(obs, deterministic=deterministic)
            return action, _states
        else:
            # If no model is loaded, return random actions
            if isinstance(obs, np.ndarray):
                # Assume action space is Discrete(7) as in TempleRunEnv
                action = np.random.randint(0, 7)
                return action, None
            else:
                # Handle other observation types if needed
                return 0, None  # Default to 'left' action

    def learn(self, total_timesteps=10000):
        """Train the model for the specified number of timesteps"""
        if self.model is not None:
            self.model.learn(total_timesteps=total_timesteps)
        else:
            print("Cannot train: No model initialized")

    def save(self, path=None):
        """Save the model to the specified path or the default path"""
        if self.model is not None:
            save_path = path if path is not None else self.model_path
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        else:
            print("Cannot save: No model initialized")
