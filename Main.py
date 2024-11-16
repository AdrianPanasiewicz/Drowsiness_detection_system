import pathlib
import time

import matplotlib.pyplot as matplot
from fastai.vision.all import *
import mediapipe as mp
import cv2
import numpy as np


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

        paramcalc = ParameterCalculator()
        coords_1 = paramcalc.find_left_eye(face_mesh_coords)
        coords_2 = paramcalc.find_right_eye(face_mesh_coords)

        fig = matplot.figure()
        ax = fig.add_subplot(projection='3d')

        for face in coords_2:
            x_list = list()
            y_list = list()
            z_list = list()

            for key,value in face.items():
                x_list.append(value.x)
                y_list.append(value.y)
                z_list.append(-value.z)

            x_list.append(x_list[0])
            y_list.append(y_list[0])
            z_list.append(z_list[0])

            x_list = np.array(x_list)
            y_list = np.array(y_list)
            z_list = np.array(z_list)

        ax.plot(x_list, z_list, y_list)

        size_xlim = np.max(x_list) - np.min(x_list)
        size_ylim = np.max(y_list) - np.min(y_list)

        ax.set(xlim=(0.45, 0.65), ylim=(-0.1, 0.1), zlim=(0.45, 0.65),
                xlabel='X', ylabel='Y', zlabel='Z')

        matplot.show()



        # output = models["Emotion_model"].predict(processed_frame_MM)
        cv2.putText(processed_frame, "Czesc", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, [17, 163, 252], 2)
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1, [17, 163, 252], 2)
        cv2.imshow('Drowsiness detection', processed_frame)
        stop_tick = time.process_time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


