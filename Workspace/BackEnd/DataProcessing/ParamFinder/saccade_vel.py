#
#                       LEGACY CODEBASE
#
# Ze względu na niezadowalające wyniki oraz niemal całkowity brak literatury naukowej
# wspierającej zastosowanie prędkości ruchów sakkadowych, podjęto decyzję o rezygnacji
# z wykorzystania tej klasy w systemie do wykrywania senności.

import time
import numpy as np
from .param_finder import ParamFinder

class SaccadeVel(ParamFinder):
    """
    Klasa odpowiedzialna za obliczanie prędkości ruchów sakkadowych oczu na podstawie
    pozycji tęczówki (iris) w kolejnych klatkach. Dodatkowo uwzględnia ruch głowy,
    aby wyliczona prędkość dotyczyła wyłącznie ruchu gałek ocznych.
    """

    def __init__(self):
        """
        Inicjalizuje obiekt SaccadeVel, definiując indeksy tęczówek (lewej i prawej)
        oraz struktury przechowujące poprzednie stany (pozycje, czas) do obliczania
        prędkości ruchu.
        """
        self.right_iris_indices = np.array([469, 470, 471, 472])
        self.left_iris_indices = np.array([474, 475, 476, 477])
        # [select_iris][index][(x, y, z, time)]
        self.iris_previous_state = np.zeros((2, 4, 4))
        # [select_iris][index][(x, y, z)]
        self.landmark_previous_state = np.zeros((2, 2, 3))

    def find_parameter(self, face_coords) -> float:
        """
        Główna metoda z interfejsu ParamFinder. Oblicza średnią prędkość (sakkady)
        na podstawie dwóch tęczówek (prawej i lewej).

        :param face_coords: Wyniki analizy MediaPipe (multi_face_landmarks).
        :type face_coords: Union[...] lub podobne
        :return: Średnia prędkość ruchów sakkadowych (float).
        :rtype: float
        """
        right_saccade_velocity = self._find_saccade_velocity(
            face_coords, self.right_iris_indices, right_flag=True
        )
        left_saccade_velocity = self._find_saccade_velocity(
            face_coords, self.left_iris_indices, left_flag=True
        )
        mean_saccade_velocity = (right_saccade_velocity + left_saccade_velocity) / 2
        return mean_saccade_velocity

    def _find_saccade_velocity(self, face_coords, indices: np.ndarray,
                               left_flag: bool = False, right_flag: bool = False) -> float:
        """
        Oblicza prędkość sakkady dla wybranej tęczówki (lewej lub prawej), korzystając
        z indeksów landmarków i poprzedniego stanu (pozycji i czasu). Umożliwia też
        kompensację ruchu głowy.

        :param face_coords: Wyniki analizy MediaPipe (multi_face_landmarks).
        :type face_coords: Union[...] lub podobne
        :param indices: Tablica indeksów tęczówki.
        :type indices: np.ndarray
        :param left_flag: Określa, czy obliczamy ruch lewej tęczówki.
        :type left_flag: bool
        :param right_flag: Określa, czy obliczamy ruch prawej tęczówki.
        :type right_flag: bool
        :return: Prędkość sakkady (float).
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
                for index, landmark in enumerate(indices):
                    velocity = self._calculate_velocity(
                        face_mesh,
                        landmark,
                        index,
                        iris_selection,
                        adjust_to_landmarks
                    )
                    all_landmark_saccade_velocity = np.append(all_landmark_saccade_velocity, velocity)

            saccade_velocity = np.mean(all_landmark_saccade_velocity)
            return saccade_velocity
        else:
            return 0.0

    def _calculate_velocity(self, face_mesh, landmark_ind: int, index: int, select_iris: int,
                           adjust_to_landmarks: list) -> float:
        """
        Oblicza prędkość (pochodną) ruchu konkretnego landmarku tęczówki między bieżącą a poprzednią
        klatką. Uwzględnia też poprawkę na ruch głowy, by zmierzyć wyłącznie ruch gałki ocznej.

        :param face_mesh: Reprezentacja pojedynczej twarzy z biblioteki MediaPipe.
        :type face_mesh: Union[...] lub podobne
        :param landmark_ind: Indeks aktualnie analizowanego landmarku tęczówki.
        :type landmark_ind: int
        :param index: Indeks w tablicy stanu (dla poszczególnych landmarków tęczówki).
        :type index: int
        :param select_iris: Indeks wskazujący, którą tęczówkę (0 - prawa, 1 - lewa) analizujemy.
        :type select_iris: int
        :param adjust_to_landmarks: Lista landmarków referencyjnych (np. kąciki oczu),
                                    względem których korygujemy ruch głowy.
        :type adjust_to_landmarks: list
        :return: Prędkość ruchu sakkady (float).
        :rtype: float
        """
        prev_state = self.iris_previous_state[select_iris][index]
        x2, y2, z2, tick2 = prev_state

        x1 = face_mesh.landmark[landmark_ind].x
        y1 = face_mesh.landmark[landmark_ind].y
        z1 = face_mesh.landmark[landmark_ind].z
        tick1 = time.time()

        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        delta_t = tick1 - tick2 if tick2 > 0 else 1e-6  # by uniknąć dzielenia przez zero

        adjust_by_distance = self.adjust_speed_to_landmarks(adjust_to_landmarks, select_iris, face_mesh)
        saccade_velocity = (distance - adjust_by_distance) / delta_t

        # Aktualizacja stanu
        self.iris_previous_state[select_iris][index] = [x1, y1, z1, tick1]

        return saccade_velocity

    def adjust_speed_to_landmarks(self, landmark_indices: list, select_iris: int, face_mesh) -> float:
        """
        Koryguje prędkość ruchu ocznego o ruch głowy. Oblicza przesunięcie
        wybranych punktów referencyjnych (np. kąciki oczu) między obecną
        a poprzednią klatką i zwraca średnią wartość tego przesunięcia.

        :param landmark_indices: Indeksy punktów referencyjnych
                                 używanych do korekcji ruchu głowy.
        :type landmark_indices: list
        :param select_iris: Wskaźnik wybierający prawą (0) lub lewą (1) tęczówkę.
        :type select_iris: int
        :param face_mesh: Obiekt opisujący bieżącą twarz (landmarki),
                          dostarczany przez MediaPipe.
        :type face_mesh: Union[...] lub podobne
        :return: Średni dystans przesunięcia punktów (float).
        :rtype: float
        """
        distances = np.zeros(len(landmark_indices))
        for idx, landmark_ind in enumerate(landmark_indices):
            prev_state = self.landmark_previous_state[select_iris][idx]
            x2, y2, z2 = prev_state

            x1 = face_mesh.landmark[landmark_ind].x
            y1 = face_mesh.landmark[landmark_ind].y
            z1 = face_mesh.landmark[landmark_ind].z

            distances[idx] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

            # Uaktualnienie stanu poprzedniego
            self.landmark_previous_state[select_iris][idx] = [x1, y1, z1]

        return distances.mean()
