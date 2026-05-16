import streamlit as st
import time
from datetime import datetime
from speech_module import SpeechEngine
from vision_system import RealVisionSystem
from stt_module import STTEngine

# Configure page
st.set_page_config(page_title="SignBridge", layout="wide")

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_text' not in st.session_state:
    st.session_state.current_text = "Waiting for gesture..."
if 'handedness' not in st.session_state:
    st.session_state.handedness = 'left'
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "Patient"
if 'last_doctor_text' not in st.session_state:
    st.session_state.last_doctor_text = ""

# Initialize backend systems (run only once)
@st.cache_resource
def get_systems():
    speech = SpeechEngine()
    vision = RealVisionSystem()
    stt = STTEngine()
    stt.start()
    return speech, vision, stt

speech_engine, vision_system, stt_engine = get_systems()

def load_css():
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
    pass

load_css()

# Sidebar for controls
with st.sidebar:
    st.title("Controls")
    
    st.markdown("### User Profile")
    handedness_option = st.radio("Handedness", ["Left Handed", "Right Handed"], index=0 if st.session_state.handedness == 'left' else 1)
    new_handedness = 'left' if handedness_option == "Left Handed" else 'right'
    if new_handedness != st.session_state.handedness:
        st.session_state.handedness = new_handedness
        vision_system.load_model(new_handedness)
        st.rerun()


            
    st.markdown("---")
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# Main Application Layout
st.title("SignBridge")

colA, colB = st.columns(2)
with colA:
    if st.button("PATIENT MODE (Sign Language)", use_container_width=True, type="primary" if st.session_state.current_mode == "Patient" else "secondary"):
        if st.session_state.current_mode != "Patient":
            st.session_state.current_mode = "Patient"
            st.rerun()
with colB:
    if st.button("DOCTOR MODE (Speech to Text)", use_container_width=True, type="primary" if st.session_state.current_mode == "Doctor" else "secondary"):
        if st.session_state.current_mode != "Doctor":
            st.session_state.current_mode = "Doctor"
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Placeholder for dynamic background styles
css_placeholder = st.empty()

col1, col2 = st.columns([2, 1])

with col1:
    doctor_subtitle_placeholder = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Live Camera Feed & Detection")
    # Placeholder for the video feed
    video_placeholder = st.empty()
    
    st.markdown("### Live Subtitles")
    # Subtitle display area
    subtitle_placeholder = st.empty()
    
with col2:
    st.subheader("Translation History")
    # History display area
    history_placeholder = st.empty()

# UI update function
def update_ui(frame, current_text, history, doctor_text="", is_emergency=False, current_mode="Patient"):
    # Turn logic
    if current_mode == "Doctor":
        doc_class = "active-turn-box"
        pat_class = "inactive-turn-box"
        cam_style = '<style>div[data-testid="stImage"] { opacity: 0.3; filter: grayscale(50%); transition: all 0.3s ease; }</style>'
    else:
        doc_class = "inactive-turn-box"
        pat_class = "active-turn-box"
        cam_style = '<style>div[data-testid="stImage"] { border: 4px solid #2dd4bf; border-radius: 12px; box-shadow: 0 0 30px rgba(45, 212, 191, 0.8); opacity: 1.0; transition: all 0.3s ease; }</style>'

    # Update Doctor Subtitles
    if doctor_text:
        doc_html = f'''
        <div class="doctor-subtitle-container {doc_class}">
            <div class="doctor-subtitle-label">Medical Personnel:</div>
            <div class="doctor-subtitle-text">{doctor_text}</div>
        </div>
        '''
    else:
        doc_html = f'''
        <div class="doctor-subtitle-container {doc_class}" style="min-height: 50px;">
            <div class="doctor-subtitle-label">Medical Personnel:</div>
            <div class="doctor-subtitle-text" style="font-size: 20px !important;">(Listening for speech...)</div>
        </div>
        '''
    doctor_subtitle_placeholder.markdown(doc_html, unsafe_allow_html=True)

    # Display video frame
    if frame is not None:
        video_placeholder.image(frame, channels="RGB", use_container_width=True)
        
    # Update Subtitle UI
    em_class = "emergency-subtitle" if is_emergency else pat_class
    
    # Update CSS
    css_string = cam_style
    if is_emergency:
        css_string += '<style>[data-testid="stAppViewContainer"] { background: #450a0a !important; }</style>'
    css_placeholder.markdown(css_string, unsafe_allow_html=True)
    
    subtitle_html = f'''
    <div class="subtitle-container {em_class}">
        <div class="subtitle-text">{current_text}</div>
    </div>
    '''
    subtitle_placeholder.markdown(subtitle_html, unsafe_allow_html=True)
    
    # Update History UI
    history_html = '<div class="history-container">'
    for item in history[:10]: # Show last 10
        history_html += f'<div class="log-entry">{item}</div>'
    history_html += '</div>'
    history_placeholder.markdown(history_html, unsafe_allow_html=True)

# Main loop
try:
    # First update to render the UI before starting the heavy loop
    update_ui(None, st.session_state.current_text, st.session_state.history, "", False, st.session_state.current_mode)
    
    EMERGENCY_GESTURES = ["help", "emergency", "pain"]
    
    while True:
        frame, gesture = vision_system.get_frame_and_gesture()
        doctor_text = stt_engine.get_latest_text()
        is_emergency = False
        
        if st.session_state.current_mode == "Patient":
            # Ignore Doctor text
            stt_engine.clear_text()
            doctor_text = ""
            
            # Process gesture
            if gesture:
                st.session_state.current_text = gesture
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                if gesture.lower() in EMERGENCY_GESTURES:
                    is_emergency = True
                    st.toast(f"EMERGENCY: {gesture.upper()} DETECTED!")
                    emergency_msg = f"URGENT ALERT: Patient requires {gesture.upper()}"
                    st.session_state.history.insert(0, f"[{timestamp}] {emergency_msg}")
                    speech_engine.say(emergency_msg, volume=1.0)
                else:
                    st.session_state.history.insert(0, f"[{timestamp}] {gesture}")
                    speech_engine.say(f"{gesture}")
                    
        elif st.session_state.current_mode == "Doctor":
            # Ignore Gesture
            gesture = None
            st.session_state.current_text = "Waiting for gesture..."
            
            # Process new STT
            if doctor_text and doctor_text != st.session_state.last_doctor_text:
                st.session_state.last_doctor_text = doctor_text
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.history.insert(0, f"[{timestamp}] Doctor: {doctor_text}")
                
        update_ui(frame, st.session_state.current_text, st.session_state.history, doctor_text, is_emergency, st.session_state.current_mode)
        
        # Small sleep to reduce CPU usage and allow Streamlit to handle sidebar events (though while loop blocks Streamlit from completing the script)
        time.sleep(0.05)
        
except Exception as e:
    st.error(f"Error: {e}")
finally:
    pass

