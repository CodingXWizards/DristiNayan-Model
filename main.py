import cv2
import numpy as np
from deepface import DeepFace
from deepface.commons import functions


# TODO: Take in different parameters from the user/request

def detection(images: str):
    text_color = (255, 255, 255)
    DeepFace.build_model('Facenet')

    print(f"Building the model {'facenet'}")

    # DeepFace.find(
    #    img_path=np.zeros([244, 244, 3]),
    #    db_path=images,
    #   model_name='Facenet',
    #    detector_backend='mediapipe',
    #   distance_metric='euclidean_l2',
    #   enforce_detection=False,
    # )

    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        frame = cv2.flip(frame, 1)
        source_obj = DeepFace.extract_faces(
            frame,
            detector_backend='mediapipe',
            enforce_detection=False)
        cv2.imshow("Frame: ", frame)
        if source_obj[0]['confidence'] > 0.8:
            x, y, w, h = source_obj[0]['facial_area'].values()
            print(x, y, w, h)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    detection('images')
