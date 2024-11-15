import pathlib
import time

from fastai.vision.all import *
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


from Workspace import *

if __name__ == "__main__":

    pathlib.PosixPath = Utils.fix_pathlib()

    # Załadowanie klasy ImageProcessor do przetwarzania obrazu
    image_processor = ImageProcessor()
    crop_size = (224,224)

    model_loader = ModelLoader()
    models = model_loader.load_models()
    # Initialise camera for video capture

    try:
        camera = cv2.VideoCapture(0)
        if camera is None or not camera.isOpened():
            raise TypeError

    except TypeError as e:
        raise TypeError('Nie udało się uzyskać dostępu do kamery lub kamera nie istnieje') from e

    past_tick = 0

    while True:
        fps = Utils.calculate_fps()

        ret, frame = camera.read()
        # processed_frame_MM = image_processor.preprocess_image1(frame, *crop_size)
        processed_frame, face_mesh_coords = image_processor.preprocess_image2(frame)

        if face_mesh_coords.multi_face_landmarks is not None:
            for face_mesh in face_mesh_coords.multi_face_landmarks:
                for i in range(len(face_mesh.landmark)):
                    print(face_mesh.landmark[i])





        # output = models["Emotion_model"].predict(processed_frame_MM)
        cv2.putText(processed_frame, "Czesc", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, [17, 163, 252], 2)
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1, [17, 163, 252], 2)
        cv2.imshow('Drowsiness detection', processed_frame)
        stop_tick = time.process_time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


