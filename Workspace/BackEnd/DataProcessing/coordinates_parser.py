import numpy as np
from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_LEFT_IRIS,
    FACEMESH_RIGHT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE
)
from Workspace.Utilities import Utils
from typing import List, Tuple, Any


class CoordinatesParser:
    """
    Klasa odpowiedzialna za przetwarzanie i wyodrębnianie współrzędnych odpowiednich
    fragmentów twarzy (np. oczy, usta, tęczówki) z wyników przetwarzania biblioteki
    MediaPipe (obiekt face_coords_results).
    """

    def __init__(self) -> None:
        """
        Inicjalizuje obiekt CoordinatesParser i przygotowuje listy indeksów
        dla poszczególnych obszarów twarzy (lewe/prawe oko, usta, tęczówki).
        """
        self._left_eye_indices = Utils.frozenset_to_list(FACEMESH_LEFT_EYE)
        self._right_eye_indices = Utils.frozenset_to_list(FACEMESH_RIGHT_EYE)
        self._mouth_indices = Utils.frozenset_to_list(FACEMESH_LIPS)
        self._left_iris_indices = Utils.frozenset_to_list(FACEMESH_LEFT_IRIS)
        self._right_iris_indices = Utils.frozenset_to_list(FACEMESH_RIGHT_IRIS)

    def find_left_eye(self, face_coords_results: Any) -> List:
        """
        Zwraca współrzędne punktów orientacyjnych lewego oka.

        :param face_coords_results: Wynik działania MediaPipe (multi_face_landmarks).
        :type face_coords_results: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkListList or similar
        :return: Lista, w której każdy element odpowiada jednej twarzy i zawiera listę „linii” oka
                 (słowników z punktami i ich współrzędnymi).
        :rtype: list
        """
        all_left_eye_coords = []
        if face_coords_results.multi_face_landmarks:
            for face_mesh in face_coords_results.multi_face_landmarks:
                left_eye_coords = []
                for line in self._left_eye_indices:
                    line_coords = {}
                    for number, index in enumerate(line, start=1):
                        line_coords[number] = face_mesh.landmark[index]
                    left_eye_coords.append(line_coords)
                all_left_eye_coords.append(left_eye_coords)
        return all_left_eye_coords

    def find_right_eye(self, face_coords_results: Any) -> List:
        """
        Zwraca współrzędne punktów orientacyjnych prawego oka.

        :param face_coords_results: Wynik działania MediaPipe (multi_face_landmarks).
        :type face_coords_results: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkListList or similar
        :return: Lista z analogiczną strukturą do find_left_eye, ale dotyczącą prawego oka.
        :rtype: list
        """
        all_right_eye_coords = []
        if face_coords_results.multi_face_landmarks:
            for face_mesh in face_coords_results.multi_face_landmarks:
                right_eye_coords = []
                for line in self._right_eye_indices:
                    line_coords = {}
                    for number, index in enumerate(line, start=1):
                        line_coords[number] = face_mesh.landmark[index]
                    right_eye_coords.append(line_coords)
                all_right_eye_coords.append(right_eye_coords)
        return all_right_eye_coords

    def find_mouth(self, face_coords_results: Any) -> List:
        """
        Zwraca współrzędne punktów orientacyjnych ust.

        :param face_coords_results: Wynik działania MediaPipe (multi_face_landmarks).
        :type face_coords_results: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkListList or similar
        :return: Lista opisująca kolejne linie ust dla każdej wykrytej twarzy.
        :rtype: list
        """
        all_mouth_coords = []
        if face_coords_results.multi_face_landmarks:
            for face_mesh in face_coords_results.multi_face_landmarks:
                mouth_coords = []
                for line in self._mouth_indices:
                    line_coords = {}
                    for number, index in enumerate(line, start=1):
                        line_coords[number] = face_mesh.landmark[index]
                    mouth_coords.append(line_coords)
                all_mouth_coords.append(mouth_coords)
        return all_mouth_coords

    @staticmethod
    def coords_to_plot_form(face_elem_coords: List) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Konwertuje współrzędne (landmarki) na trzy listy tablic NumPy zawierające współrzędne
        w osiach X, Y i Z (x_list_all, y_list_all, z_list_all).

        :param face_elem_coords: Lista współrzędnych punktów dla fragmentu twarzy (np. oka),
                                 zwykle wielopoziomowa (twarze -> linie -> słowniki).
        :type face_elem_coords: list
        :return: Krotka (x_list_all, y_list_all, z_list_all), każda to lista tablic NumPy.
        :rtype: tuple
        """
        x_list_all, y_list_all, z_list_all = [], [], []

        for face in face_elem_coords:
            x_list, y_list, z_list = [], [], []
            for line in face:
                line_x, line_y, line_z = [], [], []
                for _, value in line.items():
                    line_x.append(value.x)
                    # Odwracamy układ Y, żeby 1 oznaczało górę zamiast dół
                    line_y.append(1 - value.y)
                    line_z.append(value.z)
                x_list.append(np.array(line_x))
                y_list.append(np.array(line_y))
                z_list.append(np.array(line_z))
            x_list_all.append(np.array(x_list))
            y_list_all.append(np.array(y_list))
            z_list_all.append(np.array(z_list))

        return x_list_all, y_list_all, z_list_all

    def find_left_iris(self, face_coords_results: Any) -> List:
        """
        Zwraca współrzędne punktów orientacyjnych lewej tęczówki.

        :param face_coords_results: Wynik działania MediaPipe (multi_face_landmarks).
        :type face_coords_results: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkListList or similar
        :return: Lista opisująca linie lewej tęczówki dla każdej wykrytej twarzy.
        :rtype: list
        """
        all_left_iris_coords = []
        if face_coords_results.multi_face_landmarks:
            for face_mesh in face_coords_results.multi_face_landmarks:
                left_iris_coords = []
                for line in self._left_iris_indices:
                    line_coords = {}
                    for number, index in enumerate(line, start=1):
                        line_coords[number] = face_mesh.landmark[index]
                    left_iris_coords.append(line_coords)
                all_left_iris_coords.append(left_iris_coords)
        return all_left_iris_coords

    def find_right_iris(self, face_coords_results: Any) -> List:
        """
        Zwraca współrzędne punktów orientacyjnych prawej tęczówki.

        :param face_coords_results: Wynik działania MediaPipe (multi_face_landmarks).
        :type face_coords_results: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkListList or similar
        :return: Lista analogiczna do find_left_iris, ale dotycząca prawej tęczówki.
        :rtype: list
        """
        all_right_iris_coords = []
        if face_coords_results.multi_face_landmarks:
            for face_mesh in face_coords_results.multi_face_landmarks:
                right_iris_coords = []
                for line in self._right_iris_indices:
                    line_coords = {}
                    for number, index in enumerate(line, start=1):
                        line_coords[number] = face_mesh.landmark[index]
                    right_iris_coords.append(line_coords)
                all_right_iris_coords.append(right_iris_coords)
        return all_right_iris_coords
