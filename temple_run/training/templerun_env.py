import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import subprocess
import os
import time
from PIL import Image
import pytesseract
import hashlib
import collections
from skimage.metrics import structural_similarity as ssim
from reward_system import ImprovedRewardSystem
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from temple_run.config import RESOLUTION, OBSERVATION_SHAPE

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class TempleRunEnv(gym.Env):
    def __init__(self, frame_stack_size=4, device="cuda" if torch.cuda.is_available() else "cpu", initial_speed=0.1):
        super(TempleRunEnv, self).__init__()
        self.screen_size = (RESOLUTION, RESOLUTION)
        self.frame_stack_size = frame_stack_size
        self.observation_space = spaces.Box(low=0, high=255, shape=OBSERVATION_SHAPE, dtype=np.uint8)
        self.single_frame_shape = (self.screen_size[0], self.screen_size[1], 1)
        self.action_space = spaces.Discrete(6)
        self.actions = ['left', 'right', 'jump', 'slide', 'tilt_left', 'tilt_right']
        self.last_reward_time = time.time()
        self.start_frame_hash = None
        self.device = torch.device(device)
        self.game_speed = initial_speed
        self.last_action = -1
        self.last_frame_gray = None
        self.stuck_counter = 0
        
        self.game_initialized = False
        self.last_observation = None
        self.frame_stack = collections.deque(maxlen=self.frame_stack_size)
        self.last_raw_frame = None
        
        # Scrcpy client
        from scrcpy_client import ScrcpyClient
        self.scrcpy_client = ScrcpyClient()
        self.scrcpy_client.start()

        self.total_right_actions = 0
        self.total_wrong_actions = 0
        
        # GPU-accelerated transforms
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize(self.screen_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        
        # Debugging
        self.step_count = 0
        self.save_debug_images = False

    def _get_raw_screenshot(self):
        """Capture full-color screen image using scrcpy."""
        return self.scrcpy_client.get_frame()
    
    def _screen_changed(self, before, after, ssim_threshold=0.95):
        """Return True if screens differ significantly"""
        if before is None or after is None:
            return True # Assume change if one frame is missing
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(before_gray, after_gray, full=True)
        print(f"Screen SSIM: {score:.3f}")
        return score < ssim_threshold

    def _detect_turn(self, raw_frame):
        """Heuristically detect if a left or right turn is upcoming."""
        if raw_frame is None:
            return None

        try:
            h, w, _ = raw_frame.shape
            roi = raw_frame[h//3:h//2, :]  # Region of interest near horizon line
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            left_half = edges[:, :w//2]
            right_half = edges[:, w//2:]

            left_density = np.sum(left_half) / left_half.size
            right_density = np.sum(right_half) / right_half.size

            if left_density > right_density * 1.5:
                return 'left'
            elif right_density > left_density * 1.5:
                return 'right'
            else:
                return None
        except Exception as e:
            print(f"Turn detection failed: {e}")
            return None

    def set_speed(self, new_speed):
        """Set the game speed (delay between steps)."""
        self.game_speed = new_speed
        print(f"Game speed set to: {self.game_speed}")

    def _detect_coins(self, raw_frame):
        """Detect coins in the raw frame and identify their lane."""
        if raw_frame is None:
            return (False, False, False)
        
        h, w, _ = raw_frame.shape
        lane_width = w // 3
        
        hsv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Focus on the lower half of the screen to detect nearby coins
        mask = mask[h//2:, :]

        left_lane_mask = mask[:, :lane_width]
        center_lane_mask = mask[:, lane_width:2*lane_width]
        right_lane_mask = mask[:, 2*lane_width:]
        
        # Use a threshold to avoid noise
        coin_threshold = (h * w) * 0.001 
        
        left_coins = np.sum(left_lane_mask) > coin_threshold
        center_coins = np.sum(center_lane_mask) > coin_threshold
        right_coins = np.sum(right_lane_mask) > coin_threshold
        
        return (left_coins, center_coins, right_coins)

    def _save_debug_image(self, obs_img, prefix="debug"):
        """Save debug images to analyze classification issues"""
        if not self.save_debug_images:
            return
            
        try:
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = int(time.time())
            
            # Handle stacked frames
            if len(obs_img.shape) == 3 and obs_img.shape[2] > 1:
                for i in range(obs_img.shape[2]):
                    filename = f"{debug_dir}/{prefix}_{timestamp}_{self.step_count}_frame_{i}.png"
                    img = Image.fromarray(obs_img[:, :, i], mode='L')
                    img.save(filename)
                print(f"Saved {obs_img.shape[2]} debug frames for step {self.step_count}")
            else:
                filename = f"{debug_dir}/{prefix}_{timestamp}_{self.step_count}.png"
                if len(obs_img.shape) == 3:
                    img = Image.fromarray(obs_img.squeeze(), mode='L')
                else:
                    img = Image.fromarray(obs_img, mode='L')
                img.save(filename)
                print(f"Saved debug image: {filename}")
            
        except Exception as e:
            print(f"Failed to save debug image: {e}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.stuck_counter = 0
        
        if self.game_initialized:
            print("Player died. Waiting for game to restart automatically...")
            timeout = time.time() + 20  # 20-second timeout
            while time.time() < timeout:
                if self._find_and_click_play_button():
                    print("Clicked play button to restart game.")
                    time.sleep(3) # Wait for game to load
                    break
                time.sleep(1)
            else:
                print("Timeout waiting for game to restart. Forcing a fresh start.")
                self._start_game_fresh()

        else:
            print("First time initialization - starting Temple Run...")
            self._start_game_fresh()
            self.game_initialized = True

        obs = self._get_observation()
        if obs is None:
            obs = self._get_dummy_observation()

        # Initialize frame stack
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(obs)

        self.step_count = 0
        self.last_observation = self._get_stacked_frames()
        self.last_raw_frame = self._get_raw_screenshot()
        self._save_debug_image(self.last_observation, "reset")

        # 🟡 Print per-episode summary
        if self.step_count > 0:
            print(f"[EPISODE SUMMARY] Total Right Actions: {self.total_right_actions}, Total Wrong Actions: {self.total_wrong_actions}, Total Score: {self.total_score:.2f}")

        # 🔁 Reset totals for next episode
        self.total_right_actions = 0
        self.total_wrong_actions = 0
        self.total_score = 0.0

        return self.last_observation, {}

    def _start_game_fresh(self):
        """Force-stop the app, launch it, and wait for a visual launch confirmation."""
        print("Force-stopping Temple Run process...")
        try:
            os.system("adb shell am force-stop com.imangi.templerun")
        except Exception as e:
            print(f"Failed to force-stop, continuing...: {e}")
        
        time.sleep(1)

        print("Launching Temple Run fresh...")
        subprocess.run([
            "adb", "shell", "monkey",
            "-p", "com.imangi.templerun",
            "-c", "android.intent.category.LAUNCHER", "1"
        ], check=True)
        time.sleep(5)  # initial wait

        timeout = time.time() + 15
        while time.time() < timeout:
            if self._find_and_click_play_button():
                print("Clicked play button to start game.")
                time.sleep(3) # Wait for game to load
                return
            print("Waiting for play button...")
            time.sleep(1)

        print("Timeout elapsed waiting for play button.")

    def step(self, action):
        """Enhanced step function with context-aware reward calculation"""
        action_str = self.action_map[action]
        
        # Initialize reward system if not exists
        if not hasattr(self, 'context_reward_system'):
            from improved_reward_system import ContextAwareRewardSystem
            self.context_reward_system = ContextAwareRewardSystem()
        
        # Capture frame BEFORE action to analyze context
        pre_action_frame = self.capture_screen()
        
        # Perform action
        if action_str != 'no-op':
            self.perform_action(action_str)
        
        # Wait for action to take effect
        time.sleep(0.1)
        
        total_reward = 0
        terminated = False
        next_state = None
        
        # Process multiple frames with frame skip
        for frame_idx in range(self.frame_skip):
            # Capture frame after action
            next_state_img = self.capture_screen()
            
            if next_state_img is None:
                next_state = np.zeros((self.screen_size[0], self.screen_size[1], 3), dtype=np.uint8)
                terminated = True
                total_reward = self.context_reward_system.death_penalty
                break
            
            next_state = self.preprocess(next_state_img)
            is_alive = self.is_alive(next_state_img)
            
            if not is_alive:
                terminated = True
                break
            
            # Calculate context-aware reward using the pre-action frame
            # (because we want to evaluate if the action was appropriate for the situation)
            frame_reward = self.context_reward_system.calculate_reward(
                action_str, pre_action_frame, is_alive, terminated
            )
            
            total_reward += frame_reward
            
            # Use the latest frame for next iteration
            pre_action_frame = next_state_img
        
        # Track statistics
        self.total_reward += total_reward
        
        # Enhanced info for debugging
        info = {
            'action_taken': action_str,
            'consecutive_unnecessary': self.context_reward_system.consecutive_unnecessary_actions,
            'total_reward': self.total_reward
        }
        
        # Provide more detailed logging
        if hasattr(self.context_reward_system, 'last_obstacles'):
            detected_obstacles = [k for k, v in self.context_reward_system.last_obstacles.items() if v]
            print(f"Action: {action_str}, Reward: {total_reward:.2f}, "
                f"Terminated: {terminated}, Total: {self.total_reward:.2f}")
            if detected_obstacles:
                print(f"  Detected: {detected_obstacles}")
        
        truncated = False
        return next_state, total_reward, terminated, truncated, info

# Also add this helper method to get action suggestions (useful for debugging)
def get_action_suggestions(self):
    """Get suggested actions for current frame - useful for debugging"""
    if hasattr(self, 'context_reward_system'):
        frame = self.capture_screen()
        if frame is not None:
            suggestions, obstacles = self.context_reward_system.get_action_suggestions(frame)
            return suggestions, obstacles
    return [], {}


    def set_max_episode_steps(self, steps):
        """Dynamically update the max steps per episode for curriculum learning."""
        self.max_episode_steps = steps
        print(f"[Env] Max episode steps updated to: {steps}")


    def _get_observation(self):
        """Capture screen using scrcpy/ADB and return preprocessed RGB image."""
        raw_frame = self._get_raw_screenshot()
        if raw_frame is None:
            return None

        # Convert BGR to RGB
        rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

        # Resize to model input size (84x84 or your self.screen_size)
        resized = cv2.resize(rgb, self.screen_size, interpolation=cv2.INTER_AREA)

        # Optional sharpening per-channel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(resized, -1, kernel)

        # Return final (H, W, 3) image
        return sharpened.astype(np.uint8)

    def _perform_action(self, action):
        try:
            print(f"Performing action: {action}")
            command = []
            if action == 'left':
                command = ["adb", "shell", "input", "swipe", "800", "800", "200", "800", "100"]
            elif action == 'right':
                command = ["adb", "shell", "input", "swipe", "200", "800", "800", "800", "100"]
            elif action == 'jump':
                command = ["adb", "shell", "input", "swipe", "500", "1000", "500", "500", "100"]
            elif action == 'slide':
                command = ["adb", "shell", "input", "swipe", "500", "500", "500", "1000", "100"]
            elif action == 'tilt_left':
                subprocess.run(["adb", "shell", "sensor", "set", "acceleration", "5:0:9.81"], check=True)
                time.sleep(0.2)
                subprocess.run(["adb", "shell", "sensor", "set", "acceleration", "0:0:9.81"], check=True)
            elif action == 'tilt_right':
                subprocess.run(["adb", "shell", "sensor", "set", "acceleration", "-5:0:9.81"], check=True)
                time.sleep(0.2)
                subprocess.run(["adb", "shell", "sensor", "set", "acceleration", "0:0:9.81"], check=True)
            
            if command:
                subprocess.run(command, check=True)
        except Exception as e:
            print(f"Failed to perform action {action}: {e}")

    def _find_and_click_play_button(self):
        """Find the green play button and click it."""
        raw_frame = self._get_raw_screenshot()
        if raw_frame is None:
            return False

        hsv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 1000:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    print(f"Found potential play button at ({cx}, {cy}) with area {area}. Tapping it.")
                    subprocess.run(["adb", "shell", "input", "tap", str(cx), str(cy)], check=True)
                    return True
        return False

    def _detect_game_over_screen(self, raw_frame):
        """Use OCR to detect if the game over screen is present."""
        if raw_frame is None:
            return False
        
        try:
            h, w, _ = raw_frame.shape
            top = h // 4
            bottom = h - (h // 4)
            left = w // 4
            right = w - (w // 4)
            center_crop = raw_frame[top:bottom, left:right]

            gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray).lower()
            
            game_over_phrases = ["play again", "game over", "tap to retry", "main menu", "new high score"]
            for phrase in game_over_phrases:
                if phrase in text:
                    print(f"OCR detected game over phrase: '{phrase}' in text: '{text.strip()}'")
                    return True
        except Exception as e:
            print(f"Error during OCR: {e}")
        
        return False

    def _get_dummy_observation(self):
        return np.zeros(self.single_frame_shape, dtype=np.uint8)

    def _get_stacked_frames(self):
        return np.concatenate(list(self.frame_stack), axis=-1)

    def close(self):
        """Clean up resources when environment is closed"""
        print("Closing Temple Run environment...")
        self.scrcpy_client.stop()

    def is_swipe_action(self, action_idx):
        return self.actions[action_idx] in ['left', 'right', 'jump', 'slide']

    def is_tilt_action(self, action_idx):
        return self.actions[action_idx] in ['tilt_left', 'tilt_right']
