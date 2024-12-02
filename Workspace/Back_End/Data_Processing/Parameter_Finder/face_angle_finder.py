import math

import numpy as np

from .parameter_finder import ParameterFinder


class FaceAngleFinder(ParameterFinder):
    """Klasa odpowiedzialna za znalezienie kąta pochylenia twarzy."""

    def __init__(self):
        """Konstruktor klasy FaceAngleFinder."""
        self.face_oval_indices = np.array([[109, 148], [10, 152], [338, 377]])

    def find_parameter(self, face_coords) -> float:
        """
        Metoda do zwrócenia kątu pochylenia twarzy.

        :param face_coords: Wynik działania funkcji process od mediapipe
        :type face_coords: Union
        :return: Kąt pochylenia twarzy
        :rtype: float
        """
        face_angle = self._find_face_angle(face_coords)
        return face_angle

    def _find_face_angle(self, face_coords) -> float:
        """
        Metoda do wyliczenia kątu pochylenia twarzy.
        :param face_coords: Wynik działania funkcji process od mediapipe
        :type face_coords: Union
        :return: Kąt pochylenia twarzy
        :rtype: float
        """
        all_faces_face_angle = np.array([])

        # Dla każdej pary wskaźników na twarzy obliczenie ich pochylenia względem pionu i obliczenie średniej z nich
        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                for pair in self.face_oval_indices[0:-1]:
                    face_angle = FaceAngleFinder._calculate_angle(face_mesh, pair)
                    all_faces_face_angle = np.append(all_faces_face_angle, face_angle)

            face_angle = np.mean(all_faces_face_angle)
            return face_angle

        else:
            return 0

    @staticmethod
    def _calculate_angle(face_mesh, pair: np.array) -> float:
        """
        Metoda do wyznaczenia pochylenia dwóch wskaźników twarzy.

        :param face_mesh: Pojedyńcza twarz znaleziona przez mediapipe
        :type face_mesh: Union
        :param pair: Para wskaźników na twarzy, względem której zostanie wyliczone ich pochylenie względem pionu
        :type pair: np.array
        :return: Kąt pochylenia pary punktów względem pionu
        :rtype: float
        """
        x2 = face_mesh.landmark[pair[0]].x
        y2 = face_mesh.landmark[pair[0]].y
        z2 = face_mesh.landmark[pair[0]].z

        x1 = face_mesh.landmark[pair[1]].x
        y1 = face_mesh.landmark[pair[1]].y
        z1 = face_mesh.landmark[pair[1]].z

        delta_x = x2 - x1
        delta_y = y2 - y1
        delta_z = abs(z2 - z1)

        denominator = (delta_x * delta_x + delta_y * delta_y) ** (1 / 2)
        face_angle = math.atan2(delta_z, denominator) * 180 / math.pi * delta_y / abs(delta_y)

        return face_angle
