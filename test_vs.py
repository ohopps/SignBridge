from vision_system import RealVisionSystem
print("Init Vision System...")
vs = RealVisionSystem()
print("Getting frame...")
frame, sign = vs.get_frame_and_gesture()
print(f"Frame type: {type(frame)}")
if frame is not None:
    try:
        import numpy as np
        if hasattr(frame, 'size'):
            print(f"Frame size: {frame.size}")
            print(f"Max value: {np.max(frame)}")
            print(f"Min value: {np.min(frame)}")
    except:
        pass
vs.release()
print("Done.")
