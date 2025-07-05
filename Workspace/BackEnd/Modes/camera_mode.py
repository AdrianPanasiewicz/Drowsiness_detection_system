import cv2
import pandas as pd
from Workspace.Utilities.utils import Utils


class CameraMode:
    def __init__(self, camera, image_processor,
                 coordinates_parser, data_saver,
                 perclos_finder, yawn_finder,
                 face_tilt_finder, classifier, gui=None):
        self.camera = camera
        self.image_processor = image_processor
        self.coordinates_parser = coordinates_parser
        self.data_saver = data_saver
        self.perclos_finder = perclos_finder
        self.yawn_finder = yawn_finder
        self.face_tilt_finder = face_tilt_finder
        self.classifier = classifier
        self.gui = gui

    def run(self):
        while True:
            fps = Utils.calculate_fps()
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read camera frame.")
                break

            processed_frame, face_mesh_coords = self.image_processor.process_face_image(
                frame)

            perclos, ear = self.perclos_finder.find_parameter(
                face_mesh_coords)
            is_jawning, yawn_counter, mar = self.yawn_finder.find_parameter(
                face_mesh_coords)
            roll, pitch = self.face_tilt_finder.find_parameter(
                face_mesh_coords)

            prediction = self._calculate_prediction(
                face_mesh_coords, perclos, mar, ear, roll,
                pitch)

            if self.gui:
                self._update_gui(processed_frame,
                                 face_mesh_coords,
                                 prediction, mar,
                                 is_jawning, roll, pitch,
                                 ear, perclos, yawn_counter,
                                 fps)

            packet = {
                "MAR": mar,
                "Obecne ziewniecie": is_jawning,
                "Licznik ziewniec": yawn_counter,
                "Roll": roll,
                "Pitch": pitch,
                "EAR": ear,
                "PERCLOS": perclos,
                "Obecna sennosc": prediction,
                "FPS": fps
            }
            self.data_saver.save_to_csv(packet)

    def _calculate_prediction(self, face_mesh_coords,
                              perclos, mar, ear, roll,
                              pitch):
        if not face_mesh_coords.multi_face_landmarks:
            return None

        if perclos >= 0.25:
            return True
        elif 0.125 <= perclos < 0.25:
            data = pd.DataFrame([[mar, ear, roll, pitch]],
                                columns=["MAR", "EAR",
                                         "Roll", "Pitch"])
            return self.classifier.moving_mode_value_prediction(
                data)
        return False

    def _update_gui(self, frame, face_mesh_coords,
                    prediction, mar,
                    is_jawning, roll, pitch, ear, perclos,
                    yawn_counter, fps):
        face_plotter = self.gui.get_face_plotter()
        Utils.render_face_coordinates(
            self.coordinates_parser, face_plotter,
            face_mesh_coords)
        self.gui.set_face_plotter(face_plotter)
        self.gui.queue_parameters(prediction, mar,
                                  is_jawning, roll, pitch,
                                  ear, perclos,
                                  yawn_counter, fps)
        self.gui.queue_image(frame)