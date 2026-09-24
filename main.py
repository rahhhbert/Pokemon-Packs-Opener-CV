import cv2 as cv
import mediapipe as mp 
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Get the directory where this current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'task/hand_landmarker.task')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand marker result: {}'. format(result))
    
options = HandLandmarkerOptions( 
    base_options = BaseOptions(model_path),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = print_result,
    num_hands = 2,
    )
with HandLandmarker.create_from_options(options) as landmarker:
    print_result
    
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
    cv.imshow('Live Stream', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
    
    


    
