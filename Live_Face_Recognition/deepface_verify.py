import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import threading 
import cv2 
from deepface import DeepFace

# face_match = False
# lock = threading.lock()

import cv2
from deepface import DeepFace

def preprocess_image(img_path):
    print(f"Loading image from: {img_path}")  # Debugging statement
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Load reference image
reference_img_path = r'Live_Face_Recognition\Images\benitglasses_one.jpg'
try:
    reference_img = preprocess_image(reference_img_path)
except FileNotFoundError as e:
    print(e)
    exit(1)

# Load test image
test_img_path = r'Live_Face_Recognition\Images\benitglasses_compare.jpg'
try:
    test_img = preprocess_image(test_img_path)
except FileNotFoundError as e:
    print(e)
    exit(1)

# Verify if the test image matches the reference image
try:
    result = DeepFace.verify(test_img, reference_img, enforce_detection=False)
    print(result)  # Print result for debugging

    if result['verified']:
        print("Match")
    else:
        print("No match")
except ValueError as e:
    print(f"Error: {e}")