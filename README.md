# 🏥 SignBridge

SignBridge is an AI-powered healthcare communication assistant designed for deaf patients and healthcare personnel. It features a manual, button-based interaction system optimized for noisy hospital environments.

## Features
- **📸 Patient Mode (Sign Language):** Uses computer vision (MediaPipe) and machine learning (Random Forest) to recognize sign language gestures in real-time via the webcam and translate them into text and text-to-speech.
- **🎤 Doctor Mode (Speech to Text):** Uses speech recognition to transcribe spoken language from medical personnel into real-time, highly visible subtitles for the deaf patient.
- **🚨 Emergency Detection:** Specific gestures (like 'Help', 'Pain') automatically trigger high-contrast red UI alerts and loud auditory warnings.
- **♿ Accessibility First:** High-contrast UI, colorblind-friendly palettes, and large typography built with Streamlit.

## Prerequisites

Before running this project, anyone downloading it MUST have the following software installed on their computer:

1. **[Python (Version 3.9 or higher)](https://www.python.org/downloads/)**: When installing Python on Windows, make sure to check the box that says **"Add Python to PATH"** on the very first installation screen.
2. **A working Webcam & Microphone**: Required for gesture detection and speech recognition.

## Setup Instructions

1. **Download the Code**: Click the green `<> Code` button at the top of this GitHub page and select **"Download ZIP"**. Extract the ZIP file to a folder on your computer.
2. **Open a Terminal**: Open Command Prompt or PowerShell, and navigate to the folder where you extracted the code.
3. **Install the required libraries**: Run the following command to download all the AI models and modules required:
```bash
pip install -r requirements.txt
```

4. **Run the Application**: 
```bash
streamlit run app.py
```

## How to add custom gestures
If you want to train the AI on new hand gestures:
1. Run `python collect_data.py` to record the data via your webcam.
2. Run `python train_model.py` to retrain the Random Forest models on your new data.
3. Restart the Streamlit app.
