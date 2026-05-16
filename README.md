# 🏥 SignBridge

SignBridge is an AI-powered healthcare communication assistant designed for deaf patients and healthcare personnel. It features a manual, button-based interaction system optimized for noisy hospital environments.

## Features
- **📸 Patient Mode (Sign Language):** Uses computer vision (MediaPipe) and machine learning (Random Forest) to recognize sign language gestures in real-time via the webcam and translate them into text and text-to-speech.
- **🎤 Doctor Mode (Speech to Text):** Uses speech recognition to transcribe spoken language from medical personnel into real-time, highly visible subtitles for the deaf patient.
- **🚨 Emergency Detection:** Specific gestures (like 'Help', 'Pain') automatically trigger high-contrast red UI alerts and loud auditory warnings.
- **♿ Accessibility First:** High-contrast UI, colorblind-friendly palettes, and large typography built with Streamlit.

## Setup Instructions

1. Install the required Python libraries:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## How to add custom gestures
If you want to train the AI on new hand gestures:
1. Run `python collect_data.py` to record the data via your webcam.
2. Run `python train_model.py` to retrain the Random Forest models on your new data.
3. Restart the Streamlit app.
