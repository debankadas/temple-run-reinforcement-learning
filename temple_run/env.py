import subprocess
import cv2
import numpy as np
import os
import time
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import torch
from PIL import Image
from reward_system import ImprovedRewardSystem
from scrcpy_client import ScrcpyClient
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temple_run.config import RESOLUTION, OBSERVATION_SHAPE, FRAME_SKIP

class TempleRunEnv(gym.Env):
    def __init__(self, classifier_model, classifier_processor, agent=None, frame_skip=None):
        super(TempleRunEnv, self).__init__()
        self.scrcpy_client = ScrcpyClient()
        self.screen_size = (RESOLUTION, RESOLUTION)  # From config
        self.package_name = "com.imangi.templerun"
        self.activity_name = "com.imangi.unityactivity.ImangiUnityActivity"
        self.width = 1080
        self.height = 2400
        self.agent = agent
        self.is_first_reset = True

        self.action_space = spaces.Discrete(7)
        # Use centralized config for observation space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=OBSERVATION_SHAPE,
                                            dtype=np.uint8)
        self.action_map = {
            0: 'left',
            1: 'right',
            2: 'jump',
            3: 'slide',
            4: 'tilt_left',
            5: 'tilt_right',
            6: 'no-op'
        }
        self.classifier_model = classifier_model
        self.classifier_processor = classifier_processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_first_reset = True
        self.frame_skip = frame_skip if frame_skip is not None else FRAME_SKIP
        self.total_reward = 0
        self.total_steps = 0  # Track total steps taken
        self.arrow_template = None # Arrow detection removed
        
        # Initialize improved reward system
        self.reward_system = ImprovedRewardSystem()
        self.previous_frame = None
        
        # Track recent states and outcomes for pattern analysis
        self.recent_states = []  # Store recent (state, action, reward) tuples
        self.max_history = 10  # Keep last 10 states for comparison
        self.death_detection_buffer = [] # Buffer for consecutive dead frames
        self.consecutive_dead_frames_threshold = 1 # Reduced to 1 for immediate termination on death detection
        
        # Pre-allocate buffers to avoid memory allocation overhead
        self.empty_state = np.zeros(OBSERVATION_SHAPE, dtype=np.uint8)  # Channels-first format
        self.temp_buffer = np.empty(OBSERVATION_SHAPE, dtype=np.uint8)  # Channels-first format

    def check_adb_connection(self):
        """Checks if an ADB device is connected and ready."""
        print("Checking ADB connection...")
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, check=True, timeout=10)
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 1 and 'device' in output_lines[1]:
                print(f"ADB device found and ready: {output_lines[1].split()[0]}")
                return True
            else:
                raise ConnectionError(f"No authorized ADB device found. 'adb devices' output:\n{result.stdout}")
        except FileNotFoundError:
            raise FileNotFoundError("ADB command not found. Please ensure ADB is installed and in your system's PATH.")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise ConnectionError(f"Failed to execute 'adb devices'. Is the adb server running? Error: {e}")

    def reset(self, seed=None, options=None):
        print("[ENV] Resetting environment...")
        
        # Check ADB connection first
        try:
            # Simple ADB command to check if device is connected
            result = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=10)
            if "device" not in result.stdout:
                print("[ENV] No device connected or device is offline. Attempting to reconnect...")
                # Try to restart ADB server
                subprocess.run(["adb", "kill-server"], check=False, timeout=10)
                time.sleep(2)
                subprocess.run(["adb", "start-server"], check=False, timeout=10)
                time.sleep(3)
        except Exception as e:
            print(f"[ENV] ADB connection check failed: {e}")
        
        # Start or restart scrcpy
        if not self.scrcpy_client.is_running:
            try:
                self.scrcpy_client.start()
                time.sleep(2) # Give scrcpy time to start
            except Exception as e:
                print(f"[ENV] Failed to start scrcpy: {e}")

        try:
            # Check if Temple Run is already running
            game_running = self.is_temple_run_running()
            
            if game_running:
                # Temple Run is already running, just continue
                print("[ENV] Temple Run already running, continuing with existing session...")
                # Mark as no longer first reset since game is running
                self.is_first_reset = False
                # No action needed - just continue with the running game
            else:
                # Temple Run is not running, need to start it
                print("[ENV] Temple Run not running, starting app...")
                # Stop the app first (in case it's in a weird state)
                subprocess.run(["adb", "shell", "am", "force-stop", self.package_name], check=False, timeout=20)
                time.sleep(2)
                # Start the app
                subprocess.run(["adb", "shell", "am", "start", "-n", f"{self.package_name}/{self.activity_name}"], check=False, timeout=20)
                # Wait for the game to load completely
                time.sleep(8)  # Increased wait time for proper initialization
                self.is_first_reset = False
                
        except Exception as e:
            print(f"[ENV] Error during game reset: {e}")
        
        # Reset tracking variables
        self.total_reward = 0
        self.total_steps = 0
        self.previous_frame = None
        self.death_detection_buffer = []  # Reset death detection buffer
        
        img = self.capture_screen()
        observation = self.preprocess(img)
        info = {}
        return observation, info
    
    def is_temple_run_running(self):
        """Check if Temple Run is currently running"""
        try:
            # Check if the Temple Run process is running
            result = subprocess.run(
                ["adb", "shell", "pidof", self.package_name], 
                capture_output=True, text=True, timeout=10
            )
            
            # If pidof returns a process ID, the app is running
            if result.returncode == 0 and result.stdout.strip():
                print(f"[ENV] Temple Run is running (PID: {result.stdout.strip()})")
                return True
            else:
                print("[ENV] Temple Run is not running")
                return False
                
        except Exception as e:
            print(f"[ENV] Error checking if Temple Run is running: {e}")
            return False

    def capture_screen(self):
        # Try scrcpy first, but don't rely on it if it's not available
        if self.scrcpy_client.is_running:
            frame = self.scrcpy_client.get_frame()
            if frame is not None:
                return frame
        
        # Fallback to ADB methods
        print("[ENV] Falling back to ADB screen capture")
        return self.capture_screen_adb()

    def capture_screen_adb(self):
        # Capture screen as raw pixels (much faster than PNG)
        try:
            result = subprocess.run(['adb', 'exec-out', 'screencap'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        except subprocess.TimeoutExpired:
            print("Screen capture timed out. Is the device connected and responsive?")
            return self.capture_screen_png() # Fallback to png on timeout

        if result.returncode != 0:
            print(f"Screen capture failed: {result.stderr.decode()}")
            return None
        
        # Parse raw screencap format: header + pixel data
        raw_data = result.stdout
        if len(raw_data) < 12:
            print("Invalid screencap data")
            return None
            
        # Raw format: width(4) + height(4) + format(4) + pixel_data
        width = int.from_bytes(raw_data[0:4], byteorder='little')
        height = int.from_bytes(raw_data[4:8], byteorder='little')
        pixel_format = int.from_bytes(raw_data[8:12], byteorder='little')
        
        # Skip header and get pixel data
        pixel_data = raw_data[12:]
        
        # Most common format is RGBA_8888 (format = 1)
        if pixel_format == 1:  # RGBA_8888
            bytes_per_pixel = 4
            expected_size = width * height * bytes_per_pixel
            if len(pixel_data) >= expected_size:
                # Convert to numpy array and reshape
                img_array = np.frombuffer(pixel_data[:expected_size], dtype=np.uint8)
                img = img_array.reshape((height, width, bytes_per_pixel))
                # Convert RGBA to BGR for OpenCV compatibility
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                return img_bgr
            else:
                print(f"Pixel data size mismatch: expected {expected_size}, got {len(pixel_data)}")
                # Try to use partial data if we have enough for a small image
                if len(pixel_data) > width * 100:  # At least 100 rows of data
                    try:
                        # Calculate how many complete rows we can get
                        complete_rows = len(pixel_data) // (width * bytes_per_pixel)
                        usable_size = complete_rows * width * bytes_per_pixel
                        
                        # Reshape to the available complete rows
                        img_array = np.frombuffer(pixel_data[:usable_size], dtype=np.uint8)
                        img = img_array.reshape((complete_rows, width, bytes_per_pixel))
                        
                        # Convert RGBA to BGR for OpenCV compatibility
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                        
                        print(f"[RECOVERY] Using partial screen capture data ({complete_rows} rows)")
                        return img_bgr
                    except Exception as e:
                        print(f"[RECOVERY] Failed to use partial data: {e}")
                
                # If recovery failed, return None
                return None
        else:
            print(f"Unsupported pixel format: {pixel_format}")
            # Fallback to PNG method
            return self.capture_screen_png()
    
    def capture_screen_png(self):
        """Fallback PNG capture method"""
        try:
            result = subprocess.run(['adb', 'exec-out', 'screencap', '-p'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        except subprocess.TimeoutExpired:
            print("PNG screen capture also timed out.")
            return None

        if result.returncode != 0:
            print(f"PNG screen capture failed: {result.stderr.decode()}")
            return None
        
        image = np.frombuffer(result.stdout, dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return img

    def preprocess(self, img):
        # Convert to RGB and resize (optimized for speed)
        if img is None:
            # Return empty state in channels-first format
            return np.zeros(OBSERVATION_SHAPE, dtype=np.uint8)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to configured resolution
        resized = cv2.resize(rgb, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
        
        # Convert to channels-first format (C, H, W) for stable-baselines3
        channels_first = np.transpose(resized, (2, 0, 1))
        
        return channels_first.astype(np.uint8)

    def step(self, action):
        self.total_steps += 1  # Increment step counter
        # Print step progress every 10 steps, moved to the beginning of the step
        if self.total_steps % 10 == 0:
            print(f"[STEP COUNTER] Total steps completed: {self.total_steps}")
        
        # Convert action to int if it's a numpy array
        if isinstance(action, np.ndarray):
            action = action.item()  # Convert numpy scalar to Python int
        
        # Ensure action is an integer
        action = int(action)
        action_str = self.action_map[action]
        
        # Perform the action
        if action_str != 'no-op':
            self.perform_action(action_str)
        
        # Capture new state after action
        next_state_img = self.capture_screen()
        if next_state_img is None:
            next_state = np.zeros(OBSERVATION_SHAPE, dtype=np.uint8)  # Empty state in channels-first format
            terminated = True
            total_reward = -10.0
        else:
            next_state = self.preprocess(next_state_img)
            current_frame_is_alive = self.is_alive(next_state_img)
            
            # Handle death detection
            if not current_frame_is_alive:
                self.death_detection_buffer.append(False)
                print(f"[ENV] Death detection buffer: {len(self.death_detection_buffer)}/{self.consecutive_dead_frames_threshold}")
            else:
                self.death_detection_buffer = [] # Reset buffer if alive frame is detected
                print("[ENV] Death detection buffer reset")

            # Force termination after consecutive dead frames
            if len(self.death_detection_buffer) >= self.consecutive_dead_frames_threshold:
                terminated = True
                print(f"[ENV] Terminated after {self.consecutive_dead_frames_threshold} consecutive dead frames.")
            else:
                terminated = False
            
            # Use improved reward system for context-aware rewards
            total_reward = self.reward_system.calculate_reward(
                action=action_str,
                frame=next_state_img,  # Use full resolution frame for obstacle detection
                survived=current_frame_is_alive, # Use current frame's status for reward
                terminated=terminated # Use buffered termination status
            )
        
        self.total_reward += total_reward
        truncated = False
        info = {
            'is_alive': not terminated,
            'action_taken': action_str,
            'total_steps': self.total_steps
        }

        # Store current frame for next step's reward calculation
        if next_state_img is not None:
            self.previous_frame = next_state_img.copy()

        # Pattern analysis: compare with recent similar states
        self.analyze_pattern(next_state, action_str, total_reward)
        
        print(f"Action: {action_str}, Reward: {total_reward:.2f}, Terminated: {terminated}, Total Reward: {self.total_reward:.2f}")

        return next_state, total_reward, terminated, truncated, info

    def analyze_pattern(self, current_state, action_str, reward):
        """Analyze current state-action-reward against recent history"""
        # Create a simple state representation (mean of image regions)
        state_signature = self.create_state_signature(current_state)
        
        # Check against recent states for similar situations
        for i, (past_signature, past_action, past_reward, step_num) in enumerate(self.recent_states):
            # Calculate state similarity (simple approach)
            similarity = self.calculate_similarity(state_signature, past_signature)
            
            # If states are similar but outcomes different, print analysis
            if similarity > 0.8 and past_action != action_str and abs(past_reward - reward) > 50:
                outcome_current = "Success" if reward > 0 else "Death"
                outcome_past = "Success" if past_reward > 0 else "Death"
                
                print(f"[PATTERN] Situation {i+1}: {past_action} → {outcome_past} ({past_reward:+.0f} reward)")
                print(f"[PATTERN] Situation {len(self.recent_states)+1}: {action_str} → {outcome_current} ({reward:+.0f} reward)")
                print(f"[PATTERN] Similar states, different actions and outcomes!")
        
        # Add current state to history
        self.recent_states.append((state_signature, action_str, reward, self.total_steps))
        
        # Keep only recent history
        if len(self.recent_states) > self.max_history:
            self.recent_states.pop(0)
    
    def create_state_signature(self, state):
        """Create a simple signature for state comparison"""
        # Handle channels-first format (C, H, W)
        if len(state.shape) == 3 and state.shape[0] == 3:
            # Convert to channels-last for easier processing
            state_chw = state
            c, h, w = state_chw.shape
            # Average across channels
            state_avg = np.mean(state_chw, axis=0)
            
            # Divide image into 4 quadrants and get mean values
            quadrants = [
                float(state_avg[:h//2, :w//2].mean()),    # Top-left
                float(state_avg[:h//2, w//2:].mean()),    # Top-right
                float(state_avg[h//2:, :w//2].mean()),    # Bottom-left
                float(state_avg[h//2:, w//2:].mean()),    # Bottom-right
            ]
        else:
            # Handle channels-last format (H, W, C) or grayscale
            h, w = state.shape[:2]
            quadrants = [
                float(state[:h//2, :w//2].mean()),    # Top-left
                float(state[:h//2, w//2:].mean()),    # Top-right
                float(state[h//2:, :w//2].mean()),    # Bottom-left
                float(state[h//2:, w//2:].mean()),    # Bottom-right
            ]
        return tuple(quadrants)
    
    def calculate_similarity(self, sig1, sig2):
        """Calculate similarity between two state signatures"""
        import math
        
        # Calculate Euclidean distance and convert to similarity (0-1)
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(sig1, sig2)))
        max_distance = math.sqrt(4 * (255 ** 2))  # Maximum possible distance
        similarity = 1 - (distance / max_distance)
        return similarity

    def detect_arrow(self, img):
        return None # Arrow detection removed

    def is_alive(self, img):
        if img is None:
            print("[CLASSIFIER] No image provided, defaulting to dead state")
            return False

        # Check for the checkpoint screen first
        # if self.is_checkpoint_screen(img): # Temporarily disabled for debugging
        #     print("Checkpoint screen detected, considering it as a death event.")
        #     return False

        try:
            # Convert BGR to RGB for PIL
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Assume img is in BGR format from OpenCV
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = img
            
            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_img)
            
            # Save the current frame for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_dir = "debug_frames"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save original frame for debugging
            orig_debug_filename = f"{debug_dir}/orig_frame_{timestamp}.png"
            cv2.imwrite(orig_debug_filename, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            
            # Preprocess the image
            inputs = self.classifier_processor(images=pil_img, return_tensors="pt")
            
            # Move to device if using GPU
            if hasattr(self.classifier_model, 'device') and self.classifier_model.device != torch.device('cpu'):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get the prediction
            with torch.no_grad():
                outputs = self.classifier_model(**inputs)
                logits = outputs.logits
            
            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            alive_prob = probs[0, 0].item()  # Probability of class 0 (alive)
            dead_prob = probs[0, 1].item()   # Probability of class 1 (dead)
            
            predicted_class_idx = logits.argmax(-1).item()
            is_alive = predicted_class_idx == 0
            
            # Save frame with classification result for debugging
            debug_filename = f"{debug_dir}/frame_{timestamp}_{'alive' if is_alive else 'dead'}_{alive_prob:.2f}_{dead_prob:.2f}.png"
            cv2.imwrite(debug_filename, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            
            # Save preprocessed image as seen by the model
            processed_img = inputs['pixel_values'][0].cpu().numpy()
            # Convert from CHW to HWC format and normalize for display
            processed_img = np.transpose(processed_img, (1, 2, 0))
            processed_img = (processed_img * 255).astype(np.uint8)
            proc_debug_filename = f"{debug_dir}/proc_frame_{timestamp}_{'alive' if is_alive else 'dead'}_{alive_prob:.2f}_{dead_prob:.2f}.png"
            cv2.imwrite(proc_debug_filename, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
            
            print(f"[CLASSIFIER] Prediction: {'ALIVE' if is_alive else 'DEAD'} (Alive prob: {alive_prob:.4f}, Dead prob: {dead_prob:.4f})")
            
            # Force alive for debugging
            # return True
            
            # 0 is alive, 1 is dead
            return is_alive
            
        except Exception as e:
            print(f"[CLASSIFIER] Error in is_alive classification: {e}")
            # Save the error frame for debugging
            if img is not None:
                debug_dir = "debug_frames"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"{debug_dir}/error_frame_{timestamp}.png", img)
            return False  # Default to dead if classification fails

    def is_checkpoint_screen(self, img):
        """
        Detects if the current screen is a checkpoint respawn screen.
        This is a simple heuristic based on the presence of specific text or UI elements
        that appear after dying and respawning at a checkpoint.
        """
        # Simple color-based detection for the "Tap to continue" text or similar UI
        # This is a placeholder. A more robust method would be to use OCR or template matching.
        # For now, we'll check for a specific color in a specific region of the screen.
        # This region should be where the "Tap to continue" text appears.
        # The color should be the color of the text.
        # Let's assume the text is white and appears in the center of the screen.
        # Get the center of the screen
        h, w, _ = img.shape
        center_x, center_y = w // 2, h // 2
        # Define a region of interest (ROI) around the center
        roi = img[center_y - 50:center_y + 50, center_x - 150:center_x + 150]
        # Check for the presence of white pixels in the ROI
        # This is a very simple check and may need to be adjusted
        # based on the actual screen of the game.
        # A better approach would be to use OCR to detect the text.
        # For now, we'll just check the mean color of the ROI.
        mean_color = np.mean(roi, axis=(0, 1))
        # If the mean color is close to white, we assume it's the checkpoint screen
        if np.all(mean_color > [200, 200, 200]):
            return True
        return False

    def perform_action(self, action_str):
        print(f"Performing action: {action_str}")
        command = []
        center_x = self.width // 2
        center_y = self.height // 2
        
        try:
            if action_str == 'left':
                command = ["adb", "shell", "input", "swipe", str(center_x + 300), str(center_y), str(center_x - 300), str(center_y), "50"]
            elif action_str == 'right':
                command = ["adb", "shell", "input", "swipe", str(center_x - 300), str(center_y), str(center_x + 300), str(center_y), "50"]
            elif action_str == 'jump':
                command = ["adb", "shell", "input", "swipe", str(center_x), str(center_y + 500), str(center_x), str(center_y - 500), "50"]
            elif action_str == 'slide':
                command = ["adb", "shell", "input", "swipe", str(center_x), str(center_y - 500), str(center_x), str(center_y + 500), "50"]
            elif action_str == 'tilt_left':
                command = ["adb", "shell", "input", "keyevent", "21"]
            elif action_str == 'tilt_right':
                command = ["adb", "shell", "input", "keyevent", "22"]
            elif action_str == 'no-op':
                command = []
            
            if command:
                result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Failed to perform action {action_str}: {e}")
        except Exception as e:
            print(f"Unexpected error performing action {action_str}: {e}")

    def render(self, mode='human'):
        pass

    def close(self):
        self.scrcpy_client.stop()
