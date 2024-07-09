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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

""" Model checks camera frame based on counter, in fps """
counter = 0 
face_match = False
flag = 0 

# Load reference image
reference_images = {
    1: 'Simple_Live_Face_Recognition/Images/benit_phone_NG_1.jpg',
    2: 'Simple_Live_Face_Recognition/Images/benit_phone_G_1.jpg',
    3: 'Simple_Live_Face_Recognition/Images/basanta_webcam_1.jpg',
    4: 'Simple_Live_Face_Recognition/Images/hasana_webcam_1.jpg',
    5: 'Simple_Live_Face_Recognition/Images/ashish_webcam_1.jpg'
}

def preprocess_image(image, target_size = (224, 224)):
    """ Preprocess image by Resizing, normalizing pixel values, converting color spaces and histogram equalization """

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image 
    image_resized = cv2.resize(image_rgb, target_size)

    # Normalize pixel values
    image_normalized = image_resized / 255.0

    # Histogram equalization 
    image_equalized = cv2.equalizeHist(cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY))
    image_equalized = cv2.cvtColor(image_equalized, cv2.COLOR_GRAY2RGB)

    return image_equalized

for idx in reference_images:
    reference_images[idx] = preprocess_image(cv2.imread(reference_images[idx]))
    cv2.imwrite(f'Simple_Live_Face_Recognition/Preprocessed_Images/reference_img_{idx}.jpg', reference_images[idx])


def check_face(frame): # Machine learning function to check if user is Benit
    global face_match
    global flag
    try: # Verify using DeepFace, copy() to avoid locking scenarios
        for idx, ref_path in reference_images.items():
            if not os.path.exists(ref_path):
                print(f'Path does not exist: {ref_path}')
                continue
            ref = cv2.imread(ref_path)
            if ref is None:
                print(f'Error loading image from path: {ref_path}')
                continue
            result = DeepFace.verify(frame, ref.copy())
            if result['verified']: # and result.get('confidence', 0.0) >= 0.9: # Confidence threshold
                face_match = True # Indent Back
                flag = idx
                print(f'Face: {ref_path}, Flag: {flag}')
                return
            else:
                face_match = False
    except (ValueError, KeyError) as e:
        face_match = False
        # print(f"Error while verifying face: {e}")

while True:
    """ 
    Return value - Determine if something was returned 
    Frame to get actual captured frame
    """
    ret, frame = cap.read()

    if ret:
        # Preprocess captured frame
        preprocessed_frame = preprocess_image(frame)

        # # Save the frame as an image 
        # cv2.imwrite(f'Simple_Live_Face_Recognition/Images/captured_frame_01.jpg', preprocessed_frame)
        cv2.imwrite(f'Simple_Live_Face_Recognition/Images/captured_frame_01.jpg', frame)

        if counter % 15 == 0: # Run every 30 frames
            try: # Starting a thread
                threading.Thread(target = check_face, args = (preprocessed_frame.copy(),)).start() # Preprocessing
                # threading.Thread(target = check_face, args = (frame.copy(),)).start() # Comma put because args needs to be a tuple 

            except ValueError:
                pass
        counter += 1 # Increment counter

        """ If face is matched, display text on frame, font scale, color: BGR, thickness """
        if face_match:
            if flag == 1:
                cv2.putText(frame, "Benit with Glasses!", (20, 450), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)

            elif flag == 2:
                cv2.putText(frame, "Benit without Glasses!", (20, 450), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)

            elif flag == 3:
                cv2.putText(frame, "Basanta!", (20, 450), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)

            elif flag == 4:
                cv2.putText(frame, "Hasana!", (20, 450), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)

            elif flag == 5:
                cv2.putText(frame, "Ashish!", (20, 450), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)

        else:
            cv2.putText(frame, 'Not Recognized!', (20, 450), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0 , 255), 2)

        cv2.imshow('Live Face Recognition', frame)

    key = cv2.waitKey(1) # To recognize user key press input 
    if key == ord("q"): 
        break # Break out of loop if 'q' is pressed

cv2.destroyAllWindows() # Destroy window after loop terminates



