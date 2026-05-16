import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

handedness = input("Train model for right or left hand? (r/l): ").strip().lower()
if handedness == 'r':
    DATASET_FILE = 'gesture_dataset_right.csv'
    MODEL_FILE = 'gesture_model_right.pkl'
else:
    DATASET_FILE = 'gesture_dataset_left.csv'
    MODEL_FILE = 'gesture_model_left.pkl'

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
    9: 'Medicine',
    10: 'Headache',
    11: 'Flu',
    12: 'Cough',
    13: 'Diarrhea',
    14: 'Stomach Ache',
    15: 'Vomit',
    16: 'Fever',
    17: 'Dizzy',
    18: 'Cold',
    19: 'Weak',
    20: 'Appointment',
    21: 'Wheelchair',
    22: 'Allergy'
}

def train_model():
    if not os.path.exists(DATASET_FILE):
        print(f"Error: {DATASET_FILE} not found. Please run collect_data.py first to collect some data.")
        return

    print("Loading dataset...")
    # Read the dataset
    df = pd.read_csv(DATASET_FILE)
    
    if len(df) == 0:
        print("Dataset is empty. Please record some gestures.")
        return
        
    print(f"Total samples collected: {len(df)}")
    
    # Separate features (X) and target labels (y)
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    # Initialize the model
    # Random Forest is extremely fast and robust for this kind of coordinate data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {acc * 100:.2f}%")
    
    # Print a detailed classification report
    print("\nClassification Report:")
    # Only map labels that exist in our current dataset
    target_names = [LABEL_MAP.get(int(label), str(label)) for label in np.unique(y)]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Save the trained model to disk
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully to {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
