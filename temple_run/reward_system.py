import cv2
import numpy as np
import time
from collections import deque

class ImprovedRewardSystem:
    def __init__(self):
        # Track game state
        self.previous_frame = None
        self.position_history = deque(maxlen=10)
        self.score_history = deque(maxlen=5)
        self.last_score = 0
        self.consecutive_no_change = 0
        self.stage_progress = 0
        
        # Action necessity detection
        self.obstacle_regions = {
            'jump': (0.6, 0.8, 0.3, 0.7),  # bottom portion for logs/barriers
            'slide': (0.2, 0.4, 0.3, 0.7),  # upper portion for branches
            'left': (0.4, 0.6, 0.0, 0.3),   # left side
            'right': (0.4, 0.6, 0.7, 1.0),  # right side
        }
        
    def detect_obstacles(self, frame):
        """Detect obstacles that require specific actions"""
        if frame is None:
            return {}
        
        h, w = frame.shape[:2]
        obstacles = {}
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        for action, (y1, y2, x1, x2) in self.obstacle_regions.items():
            # Define region of interest
            y1_px, y2_px = int(y1 * h), int(y2 * h)
            x1_px, x2_px = int(x1 * w), int(x2 * w)
            
            roi = edges[y1_px:y2_px, x1_px:x2_px]
            edge_density = np.sum(roi) / roi.size
            
            # Thresholds for obstacle detection
            if action in ['jump', 'slide'] and edge_density > 0.1:
                obstacles[action] = True
            elif action in ['left', 'right'] and edge_density > 0.15:
                obstacles[action] = True
                
        return obstacles
    
    def detect_coins(self, frame):
        """Enhanced coin detection with lane information"""
        if frame is None:
            return {'left': False, 'center': False, 'right': False}
        
        h, w = frame.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Yellow color range for coins
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Focus on upcoming area (middle portion of screen)
        roi_mask = mask[h//3:2*h//3, :]
        
        # Divide into lanes
        lane_width = w // 3
        lanes = {
            'left': roi_mask[:, :lane_width],
            'center': roi_mask[:, lane_width:2*lane_width],
            'right': roi_mask[:, 2*lane_width:]
        }
        
        coin_threshold = 50  # Minimum pixels to consider as coin
        coins = {}
        for lane, mask_region in lanes.items():
            coins[lane] = np.sum(mask_region) > coin_threshold
            
        return coins
    
    def detect_stage_progress(self, frame):
        """Detect if player is making progress through stages"""
        if frame is None or self.previous_frame is None:
            return 0
        
        try:
            # Calculate optical flow to detect forward movement
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Use dense optical flow instead of sparse LK
            flow = cv2.calcOpticalFlowPyrLK(prev_gray, gray, 
                                          np.array([[frame.shape[1]//2, frame.shape[0]//2]], dtype=np.float32),
                                          None)
            
            # Analyze flow to determine forward progress
            if flow[0] is not None and len(flow[0]) > 0:
                # Simplified progress detection using single center point
                progress = flow[0][0][1] if len(flow[0][0]) > 1 else 0
                return max(0, progress * 0.1)  # Scale down the progress value
            
        except Exception as e:
            print(f"Optical flow error: {e}")
            
        return 0
    
    def extract_score_from_frame(self, frame):
        """Extract current score from game UI (simplified)"""
        # This would need OCR or template matching
        # For now, return a placeholder that increases with survival
        return self.last_score + 1
    
    def calculate_reward(self, action, frame, survived, terminated):
        """Calculate reward based on action necessity and game progress"""
        if frame is None:
            return -1.0
        
        total_reward = 0.0
        
        # Base survival reward (increased to encourage survival)
        if survived and not terminated:
            total_reward += 1  # Increased base reward for surviving
        elif terminated:
            total_reward = -10.0  # Fixed reward for termination
            
        # Detect obstacles and coins
        obstacles = self.detect_obstacles(frame)
        coins = self.detect_coins(frame)
        
        # Stage progress reward
        progress = self.detect_stage_progress(frame)
        if progress > 0:
            total_reward += progress * 0.5
            self.stage_progress += progress
        
        # Action-specific reward calculation
        if action == 'jump':
            if obstacles.get('jump', False):
                total_reward += 2.0  # Necessary jump
                print("✓ Jump: Avoided obstacle")
            else:
                total_reward -= 0.5  # Unnecessary jump
                print("✗ Jump: No obstacle detected")
                
        elif action == 'slide':
            if obstacles.get('slide', False):
                total_reward += 2.0  # Necessary slide
                print("✓ Slide: Avoided obstacle")
            else:
                total_reward -= 0.5  # Unnecessary slide
                print("✗ Slide: No obstacle detected")
                
        elif action == 'left':
            if obstacles.get('left', False):
                total_reward += 1.5  # Necessary turn
                print("✓ Left: Avoided obstacle")
            elif coins.get('left', False):
                total_reward += 1.0  # Coin collection
                print("✓ Left: Collected coins")
            else:
                total_reward -= 0.3  # Unnecessary turn
                print("✗ Left: No obstacle or coins detected")
                
        elif action == 'right':
            if obstacles.get('right', False):
                total_reward += 1.5  # Necessary turn
                print("✓ Right: Avoided obstacle")
            elif coins.get('right', False):
                total_reward += 1.0  # Coin collection
                print("✓ Right: Collected coins")
            else:
                total_reward -= 0.3  # Unnecessary turn
                print("✗ Right: No obstacle or coins detected")
                
        elif action in ['tilt_left', 'tilt_right']:
            # Tilt actions are for coin collection without changing lanes
            lane = 'left' if action == 'tilt_left' else 'right'
            if coins.get(lane, False) and not self._path_blocked(frame, lane):
                total_reward += 1.0
                print(f"✓ {action}: Collected coins while staying in lane")
            else:
                total_reward -= 0.2
                print(f"✗ {action}: No coins or path blocked")
        
        # No-op handling
        elif action == 'no-op':
            # Check if no-op was appropriate (no obstacles or coins nearby)
            if not any(obstacles.values()) and not any(coins.values()):
                total_reward += 0.1  # Small reward for appropriate no-op
            else:
                total_reward -= 1.0  # Penalty for missing necessary action
                print("✗ No-op: Action was needed")
        
        # Score-based reward
        current_score = self.extract_score_from_frame(frame)
        if current_score > self.last_score:
            score_increase = current_score - self.last_score
            total_reward += score_increase * 0.1
            print(f"✓ Score increased by {score_increase}")
        
        self.last_score = current_score
        
        # Stagnation penalty
        if self._detect_stagnation(frame):
            total_reward -= 0.5
            print("✗ Stagnation detected")
        
        # Update history
        self.previous_frame = frame.copy()
        
        return total_reward
    
    def _path_blocked(self, frame, direction):
        """Check if the path in given direction is blocked"""
        if frame is None:
            return False
        
        h, w = frame.shape[:2]
        
        # Define regions to check for obstacles
        if direction == 'left':
            roi = frame[h//2:3*h//4, :w//3]
        else:  # right
            roi = frame[h//2:3*h//4, 2*w//3:]
        
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # High edge density indicates obstacles
        edge_density = np.sum(edges) / edges.size
        return edge_density > 0.2
    
    def _detect_stagnation(self, frame):
        """Detect if the game view hasn't changed (stuck)"""
        if self.previous_frame is None:
            return False
        
        # Calculate structural similarity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Resize for faster comparison
        gray_small = cv2.resize(gray, (32, 32))
        prev_gray_small = cv2.resize(prev_gray, (32, 32))
        
        # Calculate mean squared error
        mse = np.mean((gray_small - prev_gray_small) ** 2)
        
        if mse < 10:  # Very low change
            self.consecutive_no_change += 1
        else:
            self.consecutive_no_change = 0
        
        return self.consecutive_no_change > 5
