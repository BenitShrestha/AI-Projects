"""
A live face recognition system that uses OpenCV and Deepface
"""
import threading 
import cv2 
from deepface import DeepFace

""" OpenCV camera structure,  set properties, set width, height """

# Set no. of cameras, 0 for default camera, cap -- capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set width and height of camera frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2, CAP_PROP_FRAME_HEIGHT, 480)

""" Model checks camera frame based on counter, in fps """
counter = 0 

face_match = False

# Load reference image
reference_img = cv2.imread('Live_Face_Recognition/Images/reference_img_one.jpg')

while True:
    """ 
    Return value - Determine if something was returned 
    Frame to get actual captured frame
    """
    ret, frame = cap.read()

    if ret:
        pass
    key = cv2.waitKey(1) # To recognize user key press input 
    if key == ord("q"): 
        break # Break out of loop if 'q' is pressed

cv2.destroyAllWindows() # Destroy window after loop terminates