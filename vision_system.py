import cv2
import os
import time
os.environ['GLOG_minloglevel'] = '2'
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import numpy as np
import urllib.request

# MODEL_FILE is now determined dynamically based on handedness
LANDMARKER_MODEL_PATH = 'hand_landmarker.task'

LABEL_MAP = {
    0: 'Help',
    1: 'Pain',
    2: 'Yes',
    3: 'No',
    4: 'Doctor',
    5: 'Thank You',
    6: 'Water',
    7: 'Toilet',
    8: 'Emergency',
    9: 'Medicine',
    10: 'Headache',
    11: 'Flu',
    12: 'Cough',
    13: 'Diarrhea',
    14: 'Stomach Ache',
    15: 'Vomit',
    16: 'Fever',
    17: 'Dizzy',
    19: 'Cold',
    20: 'Weak',
    22: 'Appointment',
    23: 'Wheelchair',
    25: 'Allergy'
}

class RealVisionSystem:
    def __init__(self, handedness='left'):
        self.cap = cv2.VideoCapture(0)
        
        self.handedness = handedness
        if self.handedness == 'right':
            self.model_file = 'gesture_model_right.pkl'
        else:
            self.model_file = 'gesture_model_left.pkl'
        
        # Ensure hand landmarker task exists
        if not os.path.exists(LANDMARKER_MODEL_PATH):
            print("Downloading MediaPipe Hand Landmarker model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                LANDMARKER_MODEL_PATH
            )

        # Load the trained Random Forest model
        self.load_model(self.handedness)

        # Initialize MediaPipe Hand Landmarker
        if not os.path.exists(LANDMARKER_MODEL_PATH):
            print(f"Error: {LANDMARKER_MODEL_PATH} not found.")
            self.detector = None
        else:
            base_options = python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

        from collections import deque
        self.prediction_buffer = deque(maxlen=15)
        self.last_triggered_gesture = None

    def load_model(self, handedness):
        self.handedness = handedness
        if self.handedness == 'right':
            self.model_file = 'gesture_model_right.pkl'
        else:
            self.model_file = 'gesture_model_left.pkl'
            
        if not os.path.exists(self.model_file):
            print(f"Error: {self.model_file} not found. Please train this hand's model first.")
            self.model = None
        else:
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Loaded model: {self.model_file}")

    def get_normalized_landmarks(self, hand_landmarks):
        coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
        base_x, base_y, base_z = coords[0]
        flattened = [val for x, y, z in coords for val in (x - base_x, y - base_y, z - base_z)]
        max_val = max([abs(v) for v in flattened]) if flattened else 1.0
        if max_val == 0: max_val = 1.0
        return [v / max_val for v in flattened]


    def get_frame_and_gesture(self):
        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Webcam Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, None

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)
        
        detected_sign = None

        if self.detector and self.model:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.detector.detect(mp_image)
            
            landmark_list = []
            if len(detection_result.hand_landmarks) > 0:
                h, w, _ = frame.shape
                
                # Sort hands left-to-right
                sorted_hands = sorted(detection_result.hand_landmarks, key=lambda hand: hand[0].x)
                
                for i in range(min(2, len(sorted_hands))):
                    hand_landmarks = sorted_hands[i]
                    normalized = self.get_normalized_landmarks(hand_landmarks)
                    landmark_list.extend(normalized)
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        color = (0, 255, 0) if i == 0 else (255, 0, 0)
                        cv2.circle(image_rgb, (cx, cy), 5, color, cv2.FILLED)
                        
                if len(detection_result.hand_landmarks) == 1:
                    landmark_list.extend([0.0] * 63)
                
                if len(landmark_list) == 126:
                    features = np.array(landmark_list).reshape(1, -1)
                    prediction = self.model.predict(features)[0]
                    
                    try:
                        pred_int = int(prediction)
                        current_sign = LABEL_MAP.get(pred_int, f"Unknown ({prediction})")
                    except ValueError:
                        current_sign = str(prediction)
                        
                    # More robust debounce: Majority vote over the last 15 frames
                    self.prediction_buffer.append(current_sign)
                    
                    from collections import Counter
                    counts = Counter(self.prediction_buffer)
                    most_common_sign, count = counts.most_common(1)[0]
                    
                    # If the most common sign appears in at least 60% of the buffer frames (9 out of 15)
                    if count >= 9 and len(self.prediction_buffer) == self.prediction_buffer.maxlen:
                        if most_common_sign != self.last_triggered_gesture:
                            detected_sign = most_common_sign
                            self.last_triggered_gesture = most_common_sign
                    
                    # Update display string to be the most recent raw prediction for visual feedback
                    display_sign = current_sign
                        
                    # Display prediction overlay (Modern style)
                    cv2.rectangle(image_rgb, (0, 0), (350, 60), (20, 20, 20), -1)
                    cv2.putText(image_rgb, 'PREDICTING:', (15, 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                    
                    # Highlight green if locked in (triggered), otherwise white
                    color = (0, 255, 0) if display_sign == self.last_triggered_gesture else (255, 255, 255)
                    cv2.putText(image_rgb, display_sign, (140, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
            else:
                self.last_gesture = None
                self.last_triggered_gesture = None
                self.prediction_buffer.clear()
                
        return image_rgb, detected_sign
        
    def release(self):
        if self.cap:
            self.cap.release()
