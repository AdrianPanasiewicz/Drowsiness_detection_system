import os
import cv2
import pandas as pd
import pathlib


class ImageMode:
    def __init__(self, image_processor, sql_saver,
                 perclos_finder,
                 yawn_finder, face_angle_finder,
                 classifier):
        self.image_processor = image_processor
        self.sql_saver = sql_saver
        self.perclos_finder = perclos_finder
        self.yawn_finder = yawn_finder
        self.face_angle_finder = face_angle_finder
        self.classifier = classifier
        self.label_map = {"0": "Not_drowsy", "1": "Drowsy",
                          "2": "Drowsy"}

    def process_folder(self, image_folder):
        image_paths = list(
            pathlib.Path(image_folder).glob('*.*'))
        total = len(image_paths)

        for index, image_path in enumerate(image_paths, 1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Failed to read image: {image_path}")
                continue

            self._process_image(frame, image_path)
            self._update_progress(index, total)

    def _process_image(self, frame, image_path):
        processed_frame, face_mesh_coords = self.image_processor.process_face_image(
            frame)

        if not face_mesh_coords:
            return

        perclos, ear = self.perclos_finder.find_parameter(
            face_mesh_coords)
        is_jawning, yawn_counter, mar = self.yawn_finder.find_parameter(
            face_mesh_coords)
        roll, pitch = self.face_angle_finder.find_parameter(
            face_mesh_coords)

        prediction = self._calculate_prediction(perclos,
                                                mar, ear,
                                                roll, pitch)
        drowsiness_label = self._get_label(image_path)

        packet = {
            "Obraz": str(image_path.name),
            "MAR": mar,
            "Obecne_ziewniecie": is_jawning,
            "Roll": roll,
            "Pitch": pitch,
            "EAR": ear,
            "Obecna_sennosc": drowsiness_label
        }
        self.sql_saver.save_to_csv(packet)

    def _calculate_prediction(self, perclos, mar, ear, roll,
                              pitch):
        if perclos >= 0.25:
            return True
        elif 0.125 <= perclos < 0.25:
            data = pd.DataFrame([[mar, ear, roll, pitch]],
                                columns=["MAR", "EAR",
                                         "Roll", "Pitch"])
            return self.classifier.moving_mode_value_prediction(
                data)
        return False

    def _get_label(self, image_path):
        label_path = image_path.parent.parent / 'labels' / f'{image_path.stem}.txt'
        with open(label_path, 'r') as f:
            return self.label_map[f.read(1)]

    def _update_progress(self, index, total):
        os.system('cls' if os.name == 'nt' else 'clear')
        if index % 100 == 0:
            print(f"Processed images: {index}/{total}")