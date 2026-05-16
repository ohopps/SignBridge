import numpy as np
import cv2
import time

class MockVisionSystem:
    def __init__(self):
        self.frame_count = 0
        self.mock_gestures = [None, "Help", None, "Emergency", "Doctor", None]

    def get_frame_and_gesture(self):
        # Create a dummy dark blue frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40, 30, 20)
        
        # Add some text to indicate it's a mock camera
        cv2.putText(frame, "MOCK CAMERA FEED", (150, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                    
        self.frame_count += 1
        # Change gesture every 30 frames
        gesture = self.mock_gestures[(self.frame_count // 30) % len(self.mock_gestures)]
        
        # Only return the gesture once when it transitions (like the real system's debounce)
        prev_gesture = self.mock_gestures[((self.frame_count - 1) // 30) % len(self.mock_gestures)]
        
        triggered_gesture = gesture if gesture != prev_gesture and gesture is not None else None
        
        return frame, triggered_gesture

    def release(self):
        print("MockVisionSystem released.")
