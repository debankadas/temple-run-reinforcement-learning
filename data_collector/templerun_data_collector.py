import os
import time
import subprocess
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import threading
try:
    from pynput import keyboard
except ImportError:
    print("pynput not found. Please install it: pip install pynput")
    exit()


class TempleRunDataCollector:
    def __init__(self, dataset_path="templerun_dataset"): # Corrected path
        self.dataset_path = dataset_path
        self.alive_dir = os.path.join(dataset_path, "alive")
        self.dead_dir = os.path.join(dataset_path, "dead")
        
        # Create directories
        os.makedirs(self.alive_dir, exist_ok=True)
        os.makedirs(self.dead_dir, exist_ok=True)
        
        self.alive_count = len([f for f in os.listdir(self.alive_dir) if f.endswith('.png')])
        self.dead_count = len([f for f in os.listdir(self.dead_dir) if f.endswith('.png')])
        
        print(f"Current dataset: {self.alive_count} alive, {self.dead_count} dead")

        # For on-the-fly interactive collection
        self.latest_frame = None
        self.capture_lock = threading.Lock()
        self.collecting_interactive_on_the_fly = False
    
    def capture_screen(self):
        """Capture screen using fast raw pixel method with PNG fallback"""
        try:
            # Try fast raw pixel capture first
            result = subprocess.run(['adb', 'exec-out', 'screencap'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode == 0:
                raw_data = result.stdout
                if len(raw_data) >= 12:
                    # Parse raw screencap format
                    width = int.from_bytes(raw_data[0:4], byteorder='little')
                    height = int.from_bytes(raw_data[4:8], byteorder='little')
                    pixel_format = int.from_bytes(raw_data[8:12], byteorder='little')
                    pixel_data = raw_data[12:]
                    
                    if pixel_format == 1:  # RGBA_8888
                        bytes_per_pixel = 4
                        expected_size = width * height * bytes_per_pixel
                        if len(pixel_data) >= expected_size:
                            img_array = np.frombuffer(pixel_data[:expected_size], dtype=np.uint8)
                            img = img_array.reshape((height, width, bytes_per_pixel))
                            # Convert RGBA to RGB for PIL
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                            return Image.fromarray(img_rgb)
            
            # Fallback to PNG method
            return self.capture_screen_png()
            
        except Exception as e:
            print(f"Raw capture error: {e}")
            return self.capture_screen_png()
    
    def capture_screen_png(self):
        """Fallback PNG capture method"""
        try:
            # Capture screenshot
            subprocess.run(['adb', 'shell', 'screencap', '-p', '/sdcard/temp_screen.png'], 
                         check=True, capture_output=True)
            
            # Pull screenshot
            result = subprocess.run(['adb', 'pull', '/sdcard/temp_screen.png', 'temp_capture.png'], 
                                  check=True, capture_output=True)
            
            # Clean up device
            subprocess.run(['adb', 'shell', 'rm', '/sdcard/temp_screen.png'], 
                         check=True, capture_output=True)
            
            image = Image.open('temp_capture.png')
            os.remove('temp_capture.png')  # Clean up local temp file
            return image
        
        except subprocess.CalledProcessError as e:
            print(f"Error capturing screen: {e}")
            return None
    
    def preprocess_image(self, image):
        """Basic preprocessing - resize and normalize"""
        # Resize to standard size
        image = image.resize((224, 224))
        return image
    
    def is_game_screen(self, image):
        """Basic check if this looks like a game screen"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check if image is mostly black (loading screen)
        if np.mean(img_array) < 30:
            return False
        
        # Check if image has reasonable color variation
        if np.std(img_array) < 20:
            return False
        
        return True
    
    def _on_press_interactive(self, key):
        try:
            char = key.char
            if char == 'a': # Save as alive
                with self.capture_lock:
                    if self.latest_frame:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"alive_{timestamp}.png"
                        processed_image = self.preprocess_image(self.latest_frame)
                        processed_image.save(os.path.join(self.alive_dir, filename))
                        self.alive_count += 1
                        print(f"\r✅ Saved ALIVE: {filename} (Total: A:{self.alive_count} D:{self.dead_count})", end="")
            elif char == 'd': # Save as dead
                with self.capture_lock:
                    if self.latest_frame:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"dead_{timestamp}.png"
                        processed_image = self.preprocess_image(self.latest_frame)
                        processed_image.save(os.path.join(self.dead_dir, filename))
                        self.dead_count += 1
                        print(f"\r✅ Saved DEAD: {filename} (Total: A:{self.alive_count} D:{self.dead_count})", end="")
            elif char == 'q': # Quit
                print("\nStopping interactive collection...")
                self.collecting_interactive_on_the_fly = False
                return False # Stop listener
        except AttributeError:
            # Special keys (like Shift, Ctrl, etc.) don't have a 'char' attribute
            pass

    def collect_data_interactive(self):
        """Interactive data collection - label frames on-the-fly during gameplay."""
        print("\n=== ON-THE-FLY INTERACTIVE DATA COLLECTION ===")
        print("Instructions:")
        print("  - Run this script and then switch to your Temple Run game.")
        print("  - While playing, press 'a' to save the current moment as ALIVE.")
        print("  - Press 'd' to save the current moment as DEAD.")
        print("  - Press 'q' to quit collection.")
        print("\nStarting capture... Press 'q' in this window or the game window to quit.")

        self.collecting_interactive_on_the_fly = True
        
        # Start keyboard listener in a separate thread
        listener = keyboard.Listener(on_press=self._on_press_interactive)
        listener.start()

        frames_captured_session = 0
        try:
            while self.collecting_interactive_on_the_fly:
                image = self.capture_screen()
                if image:
                    if self.is_game_screen(image): # Basic filtering
                        with self.capture_lock:
                            self.latest_frame = image.copy()
                        frames_captured_session +=1
                        if frames_captured_session % 30 == 0: # Print status every ~second if capturing at 30fps
                             print(f"\rCapturing... Press 'a' (Alive), 'd' (Dead), 'q' (Quit). Frames processed: {frames_captured_session}", end="")
                    else:
                        # print("\rSkipping non-game screen...", end="")
                        pass # Silently skip non-game screens to reduce noise
                else:
                    print("\rFailed to capture screen. Check ADB. Retrying...", end="")
                
                time.sleep(1/60) # Aim for high capture rate, adjust if CPU too high

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C).")
        finally:
            self.collecting_interactive_on_the_fly = False
            if listener.running:
                listener.stop() # Ensure listener is stopped
            listener.join() # Wait for listener thread to finish
            print(f"\n🏁 On-the-fly collection stopped.")
            print(f"Total dataset: {self.alive_count} alive, {self.dead_count} dead")
            # Remove the old interactive mode's 'auto' option or integrate if desired
            # For now, this mode replaces the old interactive one.
    
    def collect_data_auto(self):
        """Auto-collect data while you play - captures every few seconds"""
        print("\n=== AUTO COLLECTION MODE ===")
        print("This will capture screenshots every 2 seconds while you play.")
        print("After collection, you'll review and label them.")
        print("How many screenshots to collect? (recommended: 50-100)")
        
        try:
            num_screenshots = int(input("Number of screenshots: "))
        except ValueError:
            num_screenshots = 50
        
        print(f"\nStarting auto-collection of {num_screenshots} screenshots...")
        print("Start playing Temple Run now!")
        print("Press Ctrl+C to stop early")
        
        temp_images = []
        
        try:
            for i in range(num_screenshots):
                print(f"Capturing {i+1}/{num_screenshots}...", end='\r')
                
                image = self.capture_screen()
                if image and self.is_game_screen(image):
                    temp_images.append(image.copy())
                
                time.sleep(2)  # Capture every 2 seconds
        
        except KeyboardInterrupt:
            print(f"\nStopped early. Captured {len(temp_images)} screenshots.")
        
        if not temp_images:
            print("No valid screenshots captured!")
            return
        
        # Now review and label them
        print(f"\n🔍 REVIEW AND LABEL {len(temp_images)} SCREENSHOTS")
        print("Same controls: 'a' for alive, 'd' for dead, 's' to skip")
        
        for i, image in enumerate(temp_images):
            print(f"\n--- Screenshot {i+1}/{len(temp_images)} ---")
            
            # Save temp image for viewing
            temp_path = f"review_{i+1}.png"
            image.save(temp_path)
            print(f"Saved as {temp_path} for review")
            
            while True:
                choice = input("Label (a/d/s): ").lower().strip()
                
                if choice == 'a':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"alive_{timestamp}.png"
                    processed_image = self.preprocess_image(image)
                    processed_image.save(os.path.join(self.alive_dir, filename))
                    self.alive_count += 1
                    print(f"✅ Saved as ALIVE")
                    break
                
                elif choice == 'd':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"dead_{timestamp}.png"
                    processed_image = self.preprocess_image(image)
                    processed_image.save(os.path.join(self.dead_dir, filename))
                    self.dead_count += 1
                    print(f"✅ Saved as DEAD")
                    break
                
                elif choice == 's':
                    print("⏭️ Skipped")
                    break
                
                else:
                    print("Use 'a', 'd', or 's'")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        print(f"\n🎉 Auto-collection complete!")
        print(f"Total dataset: {self.alive_count} alive, {self.dead_count} dead")
    
    def analyze_dataset(self):
        """Analyze current dataset"""
        print(f"\n=== DATASET ANALYSIS ===")
        print(f"Alive images: {self.alive_count}")
        print(f"Dead images: {self.dead_count}")
        print(f"Total images: {self.alive_count + self.dead_count}")
        
        if self.alive_count + self.dead_count < 100:
            print("⚠️  WARNING: Dataset too small for good training!")
            print("   Recommended: At least 50-100 images per class")
        
        # Check balance
        if self.alive_count > 0 and self.dead_count > 0:
            ratio = max(self.alive_count, self.dead_count) / min(self.alive_count, self.dead_count)
            if ratio > 2:
                print(f"⚠️  WARNING: Dataset imbalanced (ratio {ratio:.1f}:1)")
        
        # Show sample images
        alive_files = [f for f in os.listdir(self.alive_dir) if f.endswith('.png')]
        dead_files = [f for f in os.listdir(self.dead_dir) if f.endswith('.png')]
        
        if alive_files:
            print(f"Sample alive image: {alive_files[0]}")
        if dead_files:
            print(f"Sample dead image: {dead_files[0]}")
    
    def create_dataset_json(self):
        """Create dataset configuration for training"""
        import json
        
        dataset_info = {
            "alive_count": self.alive_count,
            "dead_count": self.dead_count,
            "total_count": self.alive_count + self.dead_count,
            "alive_dir": self.alive_dir,
            "dead_dir": self.dead_dir,
            "created": datetime.now().isoformat()
        }
        
        with open(os.path.join(self.dataset_path, "dataset_info.json"), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"📋 Dataset info saved to {self.dataset_path}/dataset_info.json")

def main():
    print("🎮 TEMPLE RUN DATA COLLECTION TOOL")
    print("==================================")
    
    collector = TempleRunDataCollector()
    
    while True:
        print("\nOptions:")
        print("1. Interactive collection (label as you go)")
        print("2. Auto collection (play first, label later)")
        print("3. Analyze current dataset")
        print("4. Quit")
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice == '1':
            collector.collect_data_interactive()
        elif choice == '2':
            collector.collect_data_auto()
        elif choice == '3':
            collector.analyze_dataset()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice")
    
    # Create dataset info file
    if collector.alive_count > 0 or collector.dead_count > 0:
        collector.create_dataset_json()
        print(f"\n📊 Final dataset: {collector.alive_count} alive, {collector.dead_count} dead")
        
        if collector.alive_count + collector.dead_count >= 100:
            print("✅ Dataset size looks good for training!")
        else:
            print("⚠️  Consider collecting more data for better results")

if __name__ == "__main__":
    main()
