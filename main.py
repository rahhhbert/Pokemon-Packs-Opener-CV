import cv2 as cv
import mediapipe as mp 
import numpy as np
import os
import time



# Get the directory where this current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'task/hand_landmarker.task')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

latest_result = None

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

        
    
    
    
options = HandLandmarkerOptions( 
    base_options = BaseOptions(model_path),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = print_result,
    num_hands = 2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    min_hand_presence_confidence = 0.5)
    
    



with HandLandmarker.create_from_options(options) as hands:
    # Open camera with OpenCV
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("camera cannot be opened")
        exit()
        
        
    while True:
        success, frame = cap.read()
        if not success:
            print("cannot recieve frame .. exiting")
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        hands.detect_async(mp_frame, timestamp_ms)
        text = None
        if latest_result is not None:
            height, width, _ = frame.shape
            for hand in latest_result.handedness:
                for category in hand:
                    if category.display_name == "Right":   
                        text = "Left"
                    else:
                        text = "Right"
                        
            for hand in latest_result.hand_landmarks:
                
                for landmark in hand:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv.circle(frame, (x,y), 10, (0, 0, 255), -1)
        
        cv.putText(
        img=frame,
        text=text,
        org=(500, 500),
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=5,
        color=(0, 0, 255),
        thickness=2,
        lineType=cv.LINE_AA
    )
        cv.imshow('Window Title', frame)
        if cv.waitKey(1) == ord('q'):
            break
    




cap.release()
cv.destroyAllWindows()
    
    


    
