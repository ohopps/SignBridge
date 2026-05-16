import cv2
import os
os.environ['GLOG_minloglevel'] = '2'
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import numpy as np
import urllib.request

MODEL_FILE = 'gesture_model.pkl'
LANDMARKER_MODEL_PATH = 'hand_landmarker.task'

# Dictionary mapping numeric labels to human-readable signs
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
    9: 'Medicine'
}

def start_inference():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Please run train_model.py first.")
        return

    # Download the required task model if it doesn't exist
    if not os.path.exists(LANDMARKER_MODEL_PATH):
        print("Downloading MediaPipe Hand Landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            LANDMARKER_MODEL_PATH
        )

    print("Loading trained Random Forest model...")
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    # Initialize MediaPipe Hand Landmarker Tasks API
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    print("--- Real-Time Inference Started ---")
    print("Press 'q' to quit.")

    def get_normalized_landmarks(hand_landmarks):
        coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
        base_x, base_y, base_z = coords[0]
        flattened = [val for x, y, z in coords for val in (x - base_x, y - base_y, z - base_z)]
        max_val = max([abs(v) for v in flattened]) if flattened else 1.0
        if max_val == 0: max_val = 1.0
        return [v / max_val for v in flattened]


    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip horizontally for selfie-view
        image = cv2.flip(image, 1)
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect hands
        detection_result = detector.detect(mp_image)
        
        landmark_list = []
        
        if len(detection_result.hand_landmarks) > 0:
            h, w, _ = image.shape
            
            # Sort hands left-to-right for consistent ordering
            sorted_hands = sorted(detection_result.hand_landmarks, key=lambda hand: hand[0].x)
            for i in range(min(2, len(sorted_hands))):
                hand_landmarks = sorted_hands[i]
                # Add normalized coords to list
                normalized = get_normalized_landmarks(hand_landmarks)
                landmark_list.extend(normalized)
                
                # Draw feedback circles (green for hand 1, blue for hand 2)
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    color = (0, 255, 0) if i == 0 else (255, 0, 0)
                    cv2.circle(image, (cx, cy), 5, color, cv2.FILLED)
                    
            # Pad with zeros if only 1 hand detected
            if len(detection_result.hand_landmarks) == 1:
                landmark_list.extend([0.0] * 63)
            
            # If we have exactly 42 landmarks (126 coords)
            if len(landmark_list) == 126:
                # Reshape for sklearn prediction
                features = np.array(landmark_list).reshape(1, -1)
                prediction = model.predict(features)[0]
                
                try:
                    pred_int = int(prediction)
                    predicted_sign = LABEL_MAP.get(pred_int, f"Unknown ({prediction})")
                except ValueError:
                    predicted_sign = str(prediction)
                
                # Display the prediction on the image
                cv2.rectangle(image, (0, 0), (350, 60), (245, 117, 16), -1)
                cv2.putText(image, 'PREDICTION', (15, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, predicted_sign, (15, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
        cv2.imshow('SignHealth Gesture Recognition', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_inference()
