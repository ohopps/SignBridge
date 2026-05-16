import cv2
print("Testing Camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(f"IsOpened: {cap.isOpened()}")
ret, frame = cap.read()
print(f"Read success: {ret}")
if ret:
    print(f"Frame shape: {frame.shape}")
cap.release()
print("Done.")
