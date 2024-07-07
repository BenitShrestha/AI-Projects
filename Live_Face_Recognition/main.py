"""
A live face recognition system that uses OpenCV and Deepface
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import threading 
import cv2 
from deepface import DeepFace

""" OpenCV camera structure,  set properties, set width, height """

# Set no. of cameras, 0 for default camera, cap -- capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set width and height of camera frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

""" Model checks camera frame based on counter, in fps """
counter = 0 

face_match = False
flag = 0 

# Load reference image
reference_img = cv2.imread('Live_Face_Recognition/Images/reference_img_six.png')
second_reference_img = cv2.imread('Live_Face_Recognition/Images/reference_img_one.jpg')

def check_face(frame): # Machine learning function to check if user is Benit
    global face_match
    global flag
    try: # Verify using DeepFace, copy() to avoid locking scenarios
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
            flag = 1
        elif DeepFace.verify(frame, second_reference_img.copy())['verified']:
            face_match = True
            flag = 2
        else:
            face_match = False
    except ValueError:
        face_match = False

while True:
    """ 
    Return value - Determine if something was returned 
    Frame to get actual captured frame
    """
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0: # Run every 30 frames
            try: # Starting a thread
                threading.Thread(target = check_face, args = (frame.copy(),)).start() # Comma put because args needs to be a tuple 
            except ValueError:
                pass
        counter += 1 # Increment counter

        if face_match and flag == 1:
            """ If face is matched, display text on frame, font scale, color: BGR, thickness """
            cv2.putText(frame, "Benit no glasses!", (20, 450), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)
        elif face_match and flag == 2:
            cv2.putText(frame, "Benit with glasses!", (20, 450), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Not Benit!', (20, 450), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0 , 255), 2)

        cv2.imshow('Live Face Recognition', frame)

    key = cv2.waitKey(1) # To recognize user key press input 
    if key == ord("q"): 
        break # Break out of loop if 'q' is pressed

cv2.destroyAllWindows() # Destroy window after loop terminates


