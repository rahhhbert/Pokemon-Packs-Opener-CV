import cv2 as cv
import mediapipe as mp 
import numpy as np
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Get the directory where this current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'task/hand_landmarker.task')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result
    print('hand marker result: {}'. format(result))
    
options = HandLandmarkerOptions( 
    base_options = BaseOptions(model_path),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = print_result,
    num_hands = 2,
    )

with HandLandmarker.create_from_options(options) as landmarker:
    
    # Open camera with OpenCV
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("camera cannot be opened")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("cannot recieve frame .. exiting")
            break
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)
            
        cv.imshow('Live Stream', frame)
        if cv.waitKey(1) == ord('q'):
            break
    
    



cap.release()
cv.destroyAllWindows()
    
    


    
