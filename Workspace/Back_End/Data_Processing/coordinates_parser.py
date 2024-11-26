import numpy as np
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_FACE_OVAL , FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS
from Workspace.Utilities import Utils

class CoordinatesParser:
    def __init__(self):
        self._left_eye_indices = Utils.frozenset_to_list(FACEMESH_LEFT_EYE)
        self._right_eye_indices = Utils.frozenset_to_list(FACEMESH_RIGHT_EYE)
        self._mouth_indices = Utils.frozenset_to_list(FACEMESH_LIPS)
        self._left_iris_indices = list()
        self._right_iris_indices = list()
        self._face_oval_indices = list()

    def find_left_eye(self, results) -> list:
        """
        Metoda do zawracania współrzędnych punktów orientacyjnych lewego oka
        
        :param results: Wynik działania funkcji process od mediapipe
        :type results:
        :return: Lista z list bibliotek ze współrzędnymi punktów orientacyjnych lewego oka
        :rtype: list
        """

        all_left_eye_coords = list()
        if results.multi_face_landmarks:
            for face_mesh in results.multi_face_landmarks:
                left_eye_coords = list()
                for line in self._left_eye_indices:
                    line_coords = dict()
                    for number, index in enumerate(line,start=1):
                        line_coords.update({number:face_mesh.landmark[index]})
                    left_eye_coords.append(line_coords)

                all_left_eye_coords.append(left_eye_coords)

        return all_left_eye_coords

    def find_right_eye(self, results) -> list:
        """
        Metoda do zawracania współrzędnych punktów orientacyjnych prawego oka

        :param results: Wynik działania funkcji process od mediapipe
        :type results:
        :return: Lista z list bibliotek ze współrzędnymi punktów orientacyjnych prawego oka
        :rtype: List
        """

        all_right_eye_coords = list()
        if results.multi_face_landmarks:
            for face_mesh in results.multi_face_landmarks:
                right_eye_coords = list()
                for line in self._right_eye_indices:
                    line_coords = dict()
                    for number, index in enumerate(line,start=1):
                        line_coords.update({number:face_mesh.landmark[index]})
                    right_eye_coords.append(line_coords)

                all_right_eye_coords.append(right_eye_coords)

        return all_right_eye_coords

    def find_mouth(self, results) -> list:
        """
        Metoda do zawracania współrzędnych punktów orientacyjnych prawego oka

        :param results: Wynik działania funkcji process od mediapipe
        :type results:
        :return: Lista z list bibliotek ze współrzędnymi punktów orientacyjnych prawego oka
        :rtype: List
        """

        all_mouth_coords = list()
        if results.multi_face_landmarks:
            for face_mesh in results.multi_face_landmarks:
                mouth_coords = list()
                for line in self._mouth_indices:
                    line_coords = dict()
                    for number, index in enumerate(line,start=1):
                        line_coords.update({number:face_mesh.landmark[index]})
                    mouth_coords.append(line_coords)

                all_mouth_coords.append(mouth_coords)

        return all_mouth_coords

    @staticmethod
    def get_coordinates(right_eye_coords):
        """
        Metoda do uzyskania pozycji punktów w postaci trzech list dla osi x, y oraz z

        :param right_eye_coords:
        :type right_eye_coords:
        :return: Tuple wszystkich wymiarów
        :rtype:
        """
        x_list_all = list()
        y_list_all = list()
        z_list_all = list()

        for face in right_eye_coords:
            x_list = list()
            y_list = list()
            z_list = list()

            for line in face:
                line_x = list()
                line_y = list()
                line_z = list()
                for key, value in line.items():
                    line_x.append(value.x)
                    line_y.append(1-value.y)
                    line_z.append(value.z)

                x_list.append(line_x)
                y_list.append(line_y)
                z_list.append(line_z)

            x_list = np.array(x_list)
            y_list = np.array(y_list)
            z_list = np.array(z_list)

            x_list_all.append(x_list)
            y_list_all.append(y_list)
            z_list_all.append(z_list)

        return x_list_all, y_list_all, z_list_all