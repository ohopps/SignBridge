import cv2
import os
os.environ['GLOG_minloglevel'] = '2'
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import urllib.request

MODEL_PATH = 'hand_landmarker.task'

handedness = input("Are you right or left handed? (r/l): ").strip().lower()
if handedness == 'r':
    DATASET_FILE = 'gesture_dataset_right.csv'
else:
    DATASET_FILE = 'gesture_dataset_left.csv'

# Download the required task model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Hand Landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

# Initialize MediaPipe Hand Landmarker Tasks API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)


# Define our labels. For example:
# 0: Help, 1: Pain, 2: Yes, 3: No, 4: Doctor, 5: Thank You
print("--- Data Collection Started ---")
print("Press '0'-'9' or 'a'-'p' to save a frame for a specific gesture.")
print("0:Help, 1:Pain, 2:Yes, 3:No, 4:Doctor, 5:Thank You, 6:Water, 7:Toilet, 8:Emergency, 9:Medicine")
print("a:Headache, b:Flu, c:Cough, d:Diarrhea, e:Stomach Ache, f:Vomit, g:Fever, h:Dizzy")
print("j:Cold, k:Weak, m:Appointment, n:Wheelchair, p:Allergy")
print("Press 'q' to quit.")

# Initialize CSV file with headers if it doesn't exist
if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(42): # 42 landmarks for 2 hands (21 each)
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

def get_normalized_landmarks(hand_landmarks):
    coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
    base_x, base_y, base_z = coords[0]
    flattened = [val for x, y, z in coords for val in (x - base_x, y - base_y, z - base_z)]
    max_val = max([abs(v) for v in flattened]) if flattened else 1.0
    if max_val == 0: max_val = 1.0
    return [v / max_val for v in flattened]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("\n🚨 ERROR: Could not open your webcam! 🚨")
    print("This happens when another application is already using the camera.")
    print("If your Streamlit app (SignBridge) is currently running, it is holding the camera open.")
    print("👉 Please STOP the Streamlit app in your other terminal (Ctrl+C), or close it, and try again.\n")
    exit()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally for selfie-view
    image = cv2.flip(image, 1)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Detect hands
    detection_result = detector.detect(mp_image)
    
    landmark_list = []
    
    # Process results and draw simple circles for feedback
    if len(detection_result.hand_landmarks) > 0:
        h, w, _ = image.shape
        
        # Sort hands from left to right (by wrist x-coordinate) to ensure consistent ML features
        sorted_hands = sorted(detection_result.hand_landmarks, key=lambda hand: hand[0].x)
        
        # Extract landmarks for up to 2 hands
        for i in range(min(2, len(sorted_hands))):
            hand_landmarks = sorted_hands[i]
            # Add normalized coordinates to our list for ML
            normalized = get_normalized_landmarks(hand_landmarks)
            landmark_list.extend(normalized)
            
            # Draw circle (blue for hand 1, green for hand 2)
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                color = (255, 0, 0) if i == 0 else (0, 255, 0)
                cv2.circle(image, (cx, cy), 5, color, cv2.FILLED)
                
        # If only 1 hand is detected, pad the other 21 landmarks (63 coords) with zeros
        if len(detection_result.hand_landmarks) == 1:
            landmark_list.extend([0.0] * 63)
                
    cv2.imshow('Gesture Data Collection', image)
    
    key = cv2.waitKey(5) & 0xFF
    
    label = None
    if ord('0') <= key <= ord('9'):
        label = int(chr(key))
    elif ord('a') <= key <= ord('p'):
        label = key - ord('a') + 10
        
    if label is not None:
        if len(landmark_list) == 126:
            with open(DATASET_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                row = [label] + landmark_list
                writer.writerow(row)
            print(f"Saved gesture '{label}'")
        else:
            print("No complete hand detected to save.")
            
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
