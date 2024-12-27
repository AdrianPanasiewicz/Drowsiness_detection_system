import time

import numpy as np

from .parameter_finder import ParameterFinder


class SaccadeVelocityFinder(ParameterFinder):
    """Klasa do obliczania prędkości ruchów sakkadowych oczu."""

    def __init__(self):
        """Konstruktor klasy SaccadeVelocityFinder."""
        self.right_iris_indices = np.array([469, 470, 471, 472])
        self.left_iris_indices = np.array([474, 475, 476, 477])
        self.iris_previous_state = np.zeros((2, 4, 4))
        self.landmark_previous_state = np.zeros((2, 2, 3))

    def find_parameter(self, face_coords) -> float:
        """
        Metoda do zwracania prędkości ruchów sakkadowych oczu.

        :param face_coords: Wynik działania funkcji process od mediapipe
        :type face_coords: Union
        :return: Prędkość ruchów sakkadowych oczu
        :rtype: float
        """
        right_saccade_velocity = self._find_saccade_velocity(face_coords, self.right_iris_indices, right_flag=True)
        left_saccade_velocity = self._find_saccade_velocity(face_coords, self.left_iris_indices, left_flag=True)
        mean_saccade_velocity = (right_saccade_velocity + left_saccade_velocity) / 2

        return mean_saccade_velocity

    def _find_saccade_velocity(self, face_coords, indices: np.ndarray, left_flag: bool = False,
                               right_flag: bool = False) -> float:
        """
        Metoda do wyliczania prędkości ruchów sakkadowych oczu.

        :param face_coords: Wynik działania funkcji process od mediapipe
        :type face_coords: Union
        :param indices: Lista z indeksami wskaźników, dla których zostanie wyliczona prędkość
        :type indices: list
        :param left_flag: Flaga, czy funkcja ma obliczać prędkość ruchu sakkadowego dla lewej tęczówki
        :type left_flag: bool
        :param right_flag: Flaga, czy funkcja ma obliczać prędkość ruchu sakkadowego dla prawej tęczówki
        :type right_flag: bool
        :return: Prędkość ruchów sakkadowych oczu
        :rtype: float
        """
        all_landmark_saccade_velocity = np.array([])

        if left_flag:
            iris_selection = 1
            adjust_to_landmarks = [33, 133]
        elif right_flag:
            iris_selection = 0
            adjust_to_landmarks = [362, 263]
        else:
            iris_selection = 0
            adjust_to_landmarks = [362, 263]

        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                for index, landmark in enumerate(indices, start=0):
                    landmark_velocity = self._calculate_velocity(face_mesh, landmark, index, iris_selection,
                                                                 adjust_to_landmarks)
                    all_landmark_saccade_velocity = np.append(all_landmark_saccade_velocity, landmark_velocity)

            saccade_velocity = np.mean(all_landmark_saccade_velocity)

            return saccade_velocity
        else:
            return 0

    def _calculate_velocity(self, face_mesh, landmark_ind: tuple, index: int, select_iris: int,
                            adjust_to_landmarks: list) -> float:
        """
        Metoda do wyznaczania pochodnej z aktualnej i poprzedniej pozycji wskaźnika na twarzy

        :param face_mesh: Pojedyńcza twarz znaleziona przez mediapipe
        :type face_mesh: Union
        :param landmark_ind: Indeks do wskaźnika, dla którego zostanie obliczona prędkość
        :type landmark_ind: int
        :param index: Indeks do pamięci dla wskaźnika
        :type index: int
        :param select_iris: Zmienna do wyboru tęczówki, dla której będzie obliczana prędkość
        :type select_iris: int
        :param adjust_to_landmarks: Poprawka, aby uniezależnić prędkość ruchu sakkadowego od ruchu głowy
        :type adjust_to_landmarks: list
        :return: Prędkość
        :rtype: float
        """
        prev_state = self.iris_previous_state[select_iris][index]
        x2 = prev_state[0]
        y2 = prev_state[1]
        z2 = prev_state[2]
        tick2 = prev_state[3]

        x1 = face_mesh.landmark[landmark_ind].x
        y1 = face_mesh.landmark[landmark_ind].y
        z1 = face_mesh.landmark[landmark_ind].z
        tick1 = time.time()

        delta_x = x1 - x2
        delta_y = y1 - y2
        delta_z = z1 - z2
        delta_t = tick1 - tick2

        distance = (delta_x ** 2 + delta_y ** 2 + delta_z ** 2) ** 0.5
        adjust_by_distance = self.adjust_speed_to_landmarks(adjust_to_landmarks, select_iris, face_mesh)
        adjusted_distance = distance - adjust_by_distance

        saccade_velocity = adjusted_distance / delta_t

        self.iris_previous_state[select_iris][index][0] = x1
        self.iris_previous_state[select_iris][index][1] = y1
        self.iris_previous_state[select_iris][index][2] = z1
        self.iris_previous_state[select_iris][index][3] = tick1

        return saccade_velocity

    def adjust_speed_to_landmarks(self, landmark_indices: list, select_iris: int, face_mesh) -> float:
        """
        Metoda do uniezależnienia prędkości ruchów sakkadowych od ruchu głowy

        :param landmark_indices: Wskaźniki, względem których zostanie uniezależniona prędkość ruchu sakkadowych.
        :type landmark_indices: list
        :param select_iris: Zmienna do wyboru tęczówki, dla której będzie obliczana prędkość
        :type select_iris: int
        :param face_mesh: Pojedyńcza twarz znaleziona przez mediapipe
        :type face_mesh: Union
        :return: Dystans pokonany przez punkt
        :rtype: float
        """
        distances = np.zeros(2)
        for index, landmark_ind in enumerate(landmark_indices, start=0):
            prev_state = self.landmark_previous_state[select_iris][index]
            x2 = prev_state[0]
            y2 = prev_state[1]
            z2 = prev_state[2]

            x1 = face_mesh.landmark[landmark_ind].x
            y1 = face_mesh.landmark[landmark_ind].y
            z1 = face_mesh.landmark[landmark_ind].z

            delta_x = x1 - x2
            delta_y = y1 - y2
            delta_z = z1 - z2

            distance = (delta_x ** 2 + delta_y ** 2 + delta_z ** 2) ** 0.5

            distances[index] = distance

        adjust_by_distance = np.mean(distances)

        return adjust_by_distance
