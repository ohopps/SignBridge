from speech_module import SpeechEngine
import time

print("Starting engine")
engine = SpeechEngine()
print("Saying word 1")
engine.say("Hello")
time.sleep(2)
print("Saying word 2")
engine.say("World")
time.sleep(2)
print("Saying word 3")
engine.say("Test")
time.sleep(2)
print("Stopping")
engine.stop()
