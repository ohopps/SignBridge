import speech_recognition as sr
import threading
import time

class STTEngine:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
        # Increase energy threshold slightly for hospital environments
        self.recognizer.energy_threshold = 400
        # Dynamic energy threshold adapts to background noise
        self.recognizer.dynamic_energy_threshold = True
        
        self.latest_transcription = ""
        self.is_listening = False
        self.stop_listening_fn = None
        
        try:
            self.microphone = sr.Microphone()
            # Adjust for ambient noise briefly
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
        except Exception as e:
            print(f"[STTEngine] Microphone initialization failed: {e}")
            self.microphone = None
            
    def _callback(self, recognizer, audio):
        """Called automatically from a background thread when audio is captured."""
        try:
            text = recognizer.recognize_google(audio)
            self.latest_transcription = text
        except sr.UnknownValueError:
            pass # Speech unintelligible
        except sr.RequestError as e:
            print(f"[STTEngine] API Error: {e}")
        except Exception as e:
            print(f"[STTEngine] Unexpected Error: {e}")

    def start(self):
        if not self.is_listening and self.microphone:
            # listen_in_background spawns a daemon thread
            self.stop_listening_fn = self.recognizer.listen_in_background(self.microphone, self._callback)
            self.is_listening = True
            print("[STTEngine] Started listening to microphone.")
            
    def stop(self):
        if self.is_listening and self.stop_listening_fn:
            self.stop_listening_fn(wait_for_stop=False)
            self.is_listening = False
            
    def get_latest_text(self):
        return self.latest_transcription
        
    def clear_text(self):
        self.latest_transcription = ""
