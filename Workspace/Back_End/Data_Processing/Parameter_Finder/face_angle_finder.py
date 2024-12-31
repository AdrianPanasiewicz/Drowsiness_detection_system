import math

import numpy as np

from .parameter_finder import ParameterFinder


class FaceAngleFinder(ParameterFinder):
    """Klasa odpowiedzialna za znalezienie kąta pochylenia twarzy."""

    def __init__(self):
        """Konstruktor klasy FaceAngleFinder."""
        self.face_oval_indices = np.array([[109, 148], [10, 152], [338, 377]])

    def find_parameter(self, face_coords) -> tuple:
        """
        Metoda do zwrócenia kątu pochylenia twarzy.

        :param face_coords: Wynik działania funkcji process od mediapipe
        :type face_coords: Union
        :return: Kąt Eulera pochylenia twarzy
        :rtype: tuple
        """
        roll, pitch = self._find_face_angle(face_coords)
        return roll, pitch

    def _find_face_angle(self, face_coords) -> tuple:
        """
        Metoda do wyliczenia kątu pochylenia twarzy.
        :param face_coords: Wynik działania funkcji process od mediapipe
        :type face_coords: Union
        :return: Wybrane kąty Eulera pochylenia głowy
        :rtype: tuple
        """
        # Dla każdej pary wskaźników na twarzy obliczenie ich pochylenia względem pionu i obliczenie średniej z nich
        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                roll_single_estimates = np.array([])
                pitch_single_estimates = np.array([])
                yaw_single_estimates = np.array([])

                for pair in self.face_oval_indices[0:-1]:
                    roll, pitch = FaceAngleFinder._calculate_euler_angles(face_mesh, pair)
                    roll_single_estimates = np.append(roll_single_estimates, roll)
                    pitch_single_estimates = np.append(pitch_single_estimates, pitch)

            roll = np.mean(roll_single_estimates)
            pitch = np.mean(pitch_single_estimates)
            return roll, pitch

        else:
            return 0,0

    @staticmethod
    def _calculate_euler_angles(face_mesh, pair: np.array) -> tuple:
        """
        Metoda do wyznaczenia pochylenia dwóch wskaźników twarzy.

        :param face_mesh: Pojedyńcza twarz znaleziona przez mediapipe
        :type face_mesh: Union
        :param pair: Para wskaźników na twarzy, względem której zostanie wyliczone ich pochylenie względem pionu
        :type pair: np.array
        :return: Wybrane kąty Eulera pochylenia głowy
        :rtype: tuple
        """
        x2 = face_mesh.landmark[pair[0]].x
        y2 = face_mesh.landmark[pair[0]].y
        z2 = face_mesh.landmark[pair[0]].z

        x1 = face_mesh.landmark[pair[1]].x
        y1 = face_mesh.landmark[pair[1]].y
        z1 = face_mesh.landmark[pair[1]].z

        delta_x = x2 - x1
        delta_y = y2 - y1
        delta_z = z2 - z1

        roll = math.atan2(abs(delta_z), delta_x) * 180 / math.pi - 90
        pitch = math.atan2(delta_z, abs(delta_y)) * 180 / math.pi

        return roll, pitch
