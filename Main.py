import pathlib
import time
import os

import matplotlib.pyplot as matplot
from fastai.vision.all import *
import mediapipe as mp
import cv2
import numpy as np
import threading


from Workspace import *



if __name__ == "__main__":

    pathlib.PosixPath = Utils.fix_pathlib()

    # Załadowanie klasy ImageProcessor do przetwarzania obrazu
    image_processor = ImageProcessor()
    crop_size = (224,224)

    # model_loader = ModelLoader()
    # models = model_loader.load_models()

    parameter_calculator = CoordinatesParser()
    face_plotter = face_plotter.FacePlotter()
    os.system('cls')

    # Initialise camera for video capture
    try:
        camera = cv2.VideoCapture(0)
        if camera is None or not camera.isOpened():
            raise TypeError

    except TypeError as e:
        raise TypeError('Nie udało się uzyskać dostępu do kamery lub kamera nie istnieje') from e

    past_tick = 0

    perclos_finder = perclos_finder.PerclosFinder()


    while True:
        fps = Utils.calculate_fps()

        ret, frame = camera.read()
        # processed_frame_MM = image_processor.preprocess_image1(frame, *crop_size)
        processed_frame, face_mesh_coords = image_processor.preprocess_image2(frame)

        coords_left_eye = parameter_calculator.find_left_eye(face_mesh_coords)
        coords_right_eye = parameter_calculator.find_right_eye(face_mesh_coords)
        coords_mouth = parameter_calculator.find_mouth(face_mesh_coords)

        perclos = perclos_finder.find_parameter(face_mesh_coords)

        os.system('cls')
        print(f"PERCLOS =\t{round(perclos,2)}")

        x_list_1, y_list_1, z_list_1 = parameter_calculator.get_coordinates(coords_left_eye)
        x_list_2, y_list_2, z_list_2 = parameter_calculator.get_coordinates(coords_right_eye)
        x_list_3, y_list_3, z_list_3 = parameter_calculator.get_coordinates(coords_mouth)

        face_plotter.update_xyz_coords(x_list_1,y_list_1,z_list_1,"LEFT_EYE")
        face_plotter.update_xyz_coords(x_list_2, y_list_2, z_list_2, "RIGHT_EYE")
        face_plotter.update_xyz_coords(x_list_3, y_list_3, z_list_3, "MOUTH")

        # size_xlim = np.max(x_list_1) - np.min(x_list_1)
        # size_ylim = np.max(y_list_1) - np.min(y_list_1)

        # output = models["Emotion_model"].predict(processed_frame_MM)
        cv2.putText(processed_frame, "Emotion (Deactivated)", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, [17, 163, 252], 2)
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1, [17, 163, 252], 2)
        cv2.imshow('Drowsiness detection', processed_frame)
        stop_tick = time.process_time()

        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()



