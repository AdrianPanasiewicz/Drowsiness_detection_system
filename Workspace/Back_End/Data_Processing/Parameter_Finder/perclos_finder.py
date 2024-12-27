import numpy as np

from .parameter_finder import ParameterFinder


class PerclosFinder(ParameterFinder):
    """Klasa do obliczania parametru PERCLOS"""

    def __init__(self, perclos_threshold):
        """Konstruktor klasy PerclosFinder"""

        self.left_eye_indices = [(385, 380), (387, 373), (263, 362)]
        self.right_eye_indices = [(160, 144), (158, 153), (133, 33)]
        self.ecr_per_face_memory = {1: {1: (0, 0), 2: (0, 0)}}  # TODO należy to podmienić po debugowaniu
        self.previous_perclos = 0
        self.perclos_threshold = perclos_threshold

    def find_parameter(self, face_coords) -> float:
        """
        Metoda do zwracania parametru PERCLOS
        :param face_coords: Wynik działania funkcji process od mediapipe
        :type face_coords: Union
        :return: Parametr PERCLOS
        :rtype: float
        """

        left_ecr = self._find_eye_closure_ratio(face_coords, self.left_eye_indices)
        right_ecr = self._find_eye_closure_ratio(face_coords, self.right_eye_indices)

        if face_coords.multi_face_landmarks:
            perclos = self._calculate_perclos(left_ecr[0], right_ecr[0], 1)
            self.previous_perclos = perclos
        else:
            perclos = self.previous_perclos

        return perclos

    @staticmethod
    def _find_eye_closure_ratio(face_coords, indices: list) -> list:
        """
        Metoda do obliczania stopnia otwarcia oczu.

        :param face_coords: Wynik działania funkcji process od mediapipe
        :type face_coords: Union
        :param indices: Wartości indeksów dla wskaźników oczu
        :type indices: list
        :return: Stosunek odległości między powiekami do szerokości oczu
        :rtype: list
        """

        all_delta_ver_dist = np.array([])
        all_faces_ecr = list()

        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                for pair in indices[0:-1]:
                    y2 = face_mesh.landmark[pair[0]].y
                    x2 = face_mesh.landmark[pair[0]].x

                    y1 = face_mesh.landmark[pair[1]].y
                    x1 = face_mesh.landmark[pair[1]].x

                    delta_ver_dist = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
                    all_delta_ver_dist = np.append(all_delta_ver_dist, delta_ver_dist)

                y2 = face_mesh.landmark[indices[-1][0]].y
                x2 = face_mesh.landmark[indices[-1][0]].x

                y1 = face_mesh.landmark[indices[-1][1]].y
                x1 = face_mesh.landmark[indices[-1][1]].x
                hor_distance = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5

                all_delta_ver_dist = np.array(all_delta_ver_dist)
                mean_ver_distance = np.mean(all_delta_ver_dist)

                eye_closure_ratio = mean_ver_distance / hor_distance
                eye_closure_ratio = np.clip(eye_closure_ratio, 0, 1)
                all_faces_ecr.append(eye_closure_ratio)

        return all_faces_ecr

    def _calculate_perclos(self, left_eye_closure_ratio: float, right_eye_closure_ratio: float,
                           memory_key: int) -> float:
        """
        Metoda do wyliczania parametru PERCLOS.
        :param left_eye_closure_ratio: Stopień otwarcia lewego oka
        :type left_eye_closure_ratio: float
        :param right_eye_closure_ratio: Stopień otwarcia prawego oka
        :type right_eye_closure_ratio: float
        :param memory_key: Indeks do osoby, dla której ma być obliczony parametr PERCLOS
        :type memory_key: float
        :return: Parametr PERCLOS
        :rtype: float
        """
        period = 300  # frames
        # Usuń najstarszą klatkę i dodaj obecną, jeśli okres jest dłuższy niż 10s
        if len(self.ecr_per_face_memory[memory_key]) >= period:  # TODO Okres*FPS
            oldest_frame = min(self.ecr_per_face_memory[memory_key].keys())
            del self.ecr_per_face_memory[memory_key][oldest_frame]

        latest_frame = max(self.ecr_per_face_memory[memory_key].keys())
        ecr_ratios = (left_eye_closure_ratio, right_eye_closure_ratio)
        self.ecr_per_face_memory[memory_key].update({latest_frame + 1: ecr_ratios})

        perclos = 0
        for _, pair in self.ecr_per_face_memory[memory_key].items():
            mean_from_pair = pair[0] + pair[1]
            if mean_from_pair < self.perclos_threshold:
                perclos += 1

        perclos = perclos / period

        return perclos
