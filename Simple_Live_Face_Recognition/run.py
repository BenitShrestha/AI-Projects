from main import setup_camera, preprocess_image, check_face, preprocess_reference_images

import cv2 
import threading
from deepface import DeepFace

def main():

    # Dictionary of reference image paths
    reference_images = {
        1: 'Simple_Live_Face_Recognition/Images/benit_phone_NG_1.jpg',
        2: 'Simple_Live_Face_Recognition/Images/benit_phone_G_1.jpg',
        3: 'Simple_Live_Face_Recognition/Images/basanta_webcam_1.jpg',
        4: 'Simple_Live_Face_Recognition/Images/hasana_webcam_1.jpg',
        5: 'Simple_Live_Face_Recognition/Images/ashish_webcam_1.jpg'
    }

    cap = setup_camera() # Initialize camera
    preprocess_reference_images(reference_images) # Preprocess reference images

    # Initialize face match and flag variables
    face_match = False
    flag = None

    counter = 0
    while True:
        ret, frame = cap.read()  # Capture frame from camera
        if ret: # Indication that image was captured
            preprocessed_frame = preprocess_image(frame)  # Preprocess captured frame
            if counter % 15 == 0:  # Run face verification every 15 frames
                try:
                    # Start face verification in a thread so that operations can run concurrently with the main thread
                    # Main thread - Handles capturing frames, updating UI, while face verification executes independently
                    threading.Thread(target=check_face, args=(preprocessed_frame.copy(), reference_images)).start()  
                except ValueError:
                    pass
            counter += 1 # Raise counter by 1, Frame counting

            # Display recognition result on the frame
            if face_match:
                ''' Check for individual condtitions of reference images used '''
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
        if key == ord("q"): # Quit if 'q' key is pressed
            break
    cv2.imwrite(f'Simple_Live_Face_Recognition/Images/1_captured_frame.jpg', preprocessed_frame)  # Save captured frame
    cv2.destroyAllWindows()
    cap.release() # Release resources related to cap

if __name__ == "__main__":
    print("Run.py running")
    main()