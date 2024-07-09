import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import threading 
import cv2 
from deepface import DeepFace

# OpenCV camera setup, set properties
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)  # Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height

# Dictionary of reference image paths
reference_images = {
    1: 'Simple_Live_Face_Recognition/Images/benit_phone_NG_1.jpg',
    2: 'Simple_Live_Face_Recognition/Images/benit_phone_G_1.jpg',
    3: 'Simple_Live_Face_Recognition/Images/basanta_webcam_1.jpg',
    4: 'Simple_Live_Face_Recognition/Images/hasana_webcam_1.jpg',
    5: 'Simple_Live_Face_Recognition/Images/ashish_webcam_1.jpg'
}

# Image preprocessing - recolor, resize, normalize
def preprocess_image(image, target_size=(256, 256)):
    """ Preprocess captured frames """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, target_size)

    image_normalized = image_resized / 255.0

    # Apply histogram equalization
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)  
    image_equalized = cv2.equalizeHist(image_gray)
    
    # Convert equalized grayscale image back to RGB
    image_equalized_rgb = cv2.cvtColor(image_equalized, cv2.COLOR_GRAY2RGB)

    return image_equalized_rgb

# Preprocess reference images
for idx, ref_path in reference_images.items():
    ref_image = cv2.imread(ref_path)

    if not os.path.exists(ref_path):
        print(f"Error: File not found - {ref_path}")
        continue

    if ref_image is None:
        print(f"Error loading image from path: {ref_path}")
        continue

    else:
        img_name = ref_path[36:-4]
        print(img_name)
        preprocessed_image = preprocess_image(ref_image)
        cv2.imwrite(f'Simple_Live_Face_Recognition/Preprocessed_Images/{img_name}_preprocessed.jpg', preprocessed_image)

        reference_images[idx] = ref_image # Loads dictonary with cv2.imread(paths)
        print(f"Flag {idx}: {ref_path}")

def check_face(frame):
    global face_match, flag
    try:
        face_match = False
        for idx, ref_image in reference_images.items():
            result = DeepFace.verify(frame, ref_image.copy())  # Verify face with each reference image
            if result['verified']:
                face_match = True
                flag = idx
                print(f'Face matched with reference image at index {idx}')
                return
    except (ValueError, KeyError) as e:
        pass
        # face_match = False
        # print(f"Error while verifying face: {e}")

counter = 0
while True:
    ret, frame = cap.read()  # Capture frame from camera
    if ret:
        preprocessed_frame = preprocess_image(frame)  # Preprocess captured frame
        if counter % 15 == 0:  # Run face verification every 15 frames
            try:
                threading.Thread(target=check_face, args=(preprocessed_frame.copy(),)).start()  # Start face verification in a thread
            except ValueError:
                pass
        counter += 1

        # Display recognition result on the frame
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

        cv2.imshow('Live Face Recognition', frame)  # Display frame with face recognition overlay

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.imwrite(f'Simple_Live_Face_Recognition/Images/1_captured_frame.jpg', preprocessed_frame)  # Save captured frame
cv2.destroyAllWindows()
cap.release()