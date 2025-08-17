import subprocess
import threading
import numpy as np
import cv2
import shutil
import re
import time

class ScrcpyClient:
    def __init__(self, device=None, max_width=800):
        self.device = device
        self.max_width = max_width
        self.process = None
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None
        self.is_running = False
        self.width = 0
        self.height = 0

    def _check_ffmpeg(self):
        """Checks if ffmpeg is in the system's PATH."""
        return shutil.which("ffmpeg") is not None

    def _get_device_resolution(self):
        """Gets the device resolution using ADB and calculates the scaled size."""
        try:
            command = ["adb", "shell", "wm", "size"]
            if self.device:
                command.extend(["-s", self.device])
            
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
            
            match = re.search(r'(\d+)x(\d+)', result.stdout)
            if match:
                original_width = int(match.group(1))
                original_height = int(match.group(2))
                
                if original_width > self.max_width:
                    self.height = int(original_height * self.max_width / original_width)
                    self.width = self.max_width
                else:
                    self.width = original_width
                    self.height = original_height
                
                # Ensure height is even for some video codecs
                if self.height % 2 != 0:
                    self.height -= 1

                print(f"[INFO] Device resolution: {original_width}x{original_height}, Scaled to: {self.width}x{self.height}")
                return True
            else:
                print("[ERROR] Could not parse device resolution from adb.")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to get device resolution: {e}")
            return False

    def _start_scrcpy(self):
        print("[DIAGNOSTIC] Executing _start_scrcpy from version with MKV recording logic (v20250625_1945)") # Unique diagnostic print
        command = [
            "scrcpy",
            "--record", "-",  # Record to stdout
            "--record-format", "mkv", # Use MKV container
            "--stay-awake",
            f"--max-size={self.max_width}",
            "--no-audio" # Optional: disable audio if not needed
        ]
        if self.device:
            command.extend(["--serial", self.device])

        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Start a thread to log scrcpy's stderr
        def log_scrcpy_stderr():
            if self.process:
                for line in iter(self.process.stderr.readline, b''):
                    print(f"[scrcpy-error] {line.decode().strip()}")

        scrcpy_stderr_thread = threading.Thread(target=log_scrcpy_stderr)
        scrcpy_stderr_thread.daemon = True
        scrcpy_stderr_thread.start()

    def _read_stream(self):
        # No header to consume with --record format

        ffmpeg_command = [
            "ffmpeg",
            "-loglevel", "error", # Reduce ffmpeg's own console noise, we have stderr logger
            "-f", "matroska", # Explicitly state input format
            "-ignore_unknown", # Ignore any non-video data scrcpy might send first
            "-probesize", "32",
            "-analyzeduration", "0",
            "-fflags", "nobuffer",
            "-i", "pipe:0",
            "-r", "60",
            "-s", f"{self.width}x{self.height}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]
        
        try:
            self.ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=self.process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"[ERROR] Failed to start ffmpeg process: {e}")
            return

        def log_ffmpeg_stderr():
            for line in iter(self.ffmpeg_process.stderr.readline, b''):
                print(f"[ffmpeg-error] {line.decode().strip()}")
        
        ffmpeg_stderr_thread = threading.Thread(target=log_ffmpeg_stderr)
        ffmpeg_stderr_thread.daemon = True
        ffmpeg_stderr_thread.start()

        while self.is_running:
            frame_size = self.width * self.height * 3
            in_bytes = self.ffmpeg_process.stdout.read(frame_size)
            
            if not in_bytes or len(in_bytes) != frame_size:
                break
            
            frame = np.frombuffer(in_bytes, np.uint8).reshape((self.height, self.width, 3))
            with self.lock:
                self.frame = frame

    def start(self):
        if not self._check_ffmpeg():
            print("[WARNING] ffmpeg not found. scrcpy client will not start.")
            return

        # Try up to 3 times to connect
        for attempt in range(3):
            if attempt > 0:
                print(f"[RECONNECT] Attempt {attempt+1}/3 to start scrcpy client...")
                time.sleep(2)  # Wait before retry
            
            # Check ADB connection first
            try:
                result = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True, timeout=10)
                if "device" not in result.stdout:
                    print("[ERROR] No connected devices found. Retrying...")
                    continue
            except Exception as e:
                print(f"[ERROR] ADB connection check failed: {e}")
                continue
                
            if not self._get_device_resolution():
                print("[ERROR] Could not get device resolution. Retrying...")
                continue

            self._start_scrcpy()
            if not self.process:  # Check if scrcpy process failed to start
                print("[ERROR] scrcpy process did not start. Retrying...")
                continue

            # Start the frame reading thread
            self.thread = threading.Thread(target=self._read_stream, daemon=True)
            self.thread.start()
            self.is_running = True
            print("[SUCCESS] scrcpy client started successfully")
            return  # Success, exit the retry loop
            
        # If we get here, all attempts failed
        print("[CRITICAL] Failed to start scrcpy client after 3 attempts")

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            else:
                # Check if process is still running
                if self.process and self.process.poll() is not None:
                    print("[WARNING] scrcpy process has terminated unexpectedly")
                    self.is_running = False
                    # Try to restart
                    self.start()
                return None

    def stop(self):
        self.is_running = False
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
            self.ffmpeg_process.terminate()
        if self.process:
            self.process.terminate()
        if self.thread:
            self.thread.join()
