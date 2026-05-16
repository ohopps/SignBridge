import pyttsx3
import threading
import queue

class SpeechEngine:
    def __init__(self):
        # We use a queue and a background thread to prevent TTS from blocking the main camera loop
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        # Initialize pyttsx3 in the thread where it will be used
        # Pythoncom initialization might be needed for Windows COM objects in threads,
        # but usually pyttsx3 handles this if init is called inside the thread
        import pythoncom
        pythoncom.CoInitialize()
        
        while True:
            item = self.q.get()
            if item is None:
                break
            
            text, volume = item
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                if len(voices) > 1:
                    engine.setProperty('voice', voices[1].id)
                engine.setProperty('rate', 150)
                engine.setProperty('volume', volume)
                engine.say(text)
                engine.runAndWait()
                # Clean up to avoid deadlocks
                del engine
            except Exception as e:
                print(f"[SpeechEngine] Error: {e}")
            finally:
                self.q.task_done()

    def say(self, text, volume=0.9):
        """Adds text to the speech queue to be spoken asynchronously."""
        self.q.put((text, volume))
        
    def stop(self):
        """Stops the speech engine thread."""
        self.q.put(None)
        self.thread.join()
