import numpy as np

class ParameterCalculator:
    def __init__(self):
        self._left_eye_indices = [398,384,385,386,387,388,466,263,249,390,373,374,380,381]
        self._right_eye_indices = [7,163,144,145,153,154,173,157,158,159,160,161]
        self._mouth_indices = [185,40,39,37,0,267,269,270,375,321,405,314,17,84,181,91,146,185,178,14,312,402]
        self._iris_indices = list()

    def find_left_eye(self, results) -> list:
        """
        Metoda do zawracania współrzędnych punktów orientacyjnych lewego oka
        
        :param results: Wynik działania funkcji process od mediapipe
        :type results:
        :return: Lista bibliotek ze współrzędnymi punktów orientacyjnych lewego oka
        :rtype: List
        """

        all_left_eye_coords = list()
        if results.multi_face_landmarks:
            for face_mesh in results.multi_face_landmarks:
                left_eye_coords = dict()
                for number, index in enumerate(self._left_eye_indices,start=1):
                    left_eye_coords.update({number:face_mesh.landmark[index]})
                    
                all_left_eye_coords.append(left_eye_coords)

        return all_left_eye_coords

    def find_right_eye(self, results) -> list:
        """
        Metoda do zawracania współrzędnych punktów orientacyjnych prawego oka

        :param results: Wynik działania funkcji process od mediapipe
        :type results:
        :return: Lista bibliotek ze współrzędnymi punktów orientacyjnych prawego oka
        :rtype: List
        """

        all_right_eye_coords = list()
        if results.multi_face_landmarks:
            for face_mesh in results.multi_face_landmarks:
                right_eye_coords = dict()
                for number,index in enumerate(self._right_eye_indices,start = 1):
                    right_eye_coords.update({number:face_mesh.landmark[index]})

                all_right_eye_coords.append(right_eye_coords)

        return all_right_eye_coords

    def find_mouth(self, results) -> list:
        """
        Metoda do zawracania współrzędnych punktów orientacyjnych prawego oka

        :param results: Wynik działania funkcji process od mediapipe
        :type results:
        :return: Lista bibliotek ze współrzędnymi punktów orientacyjnych prawego oka
        :rtype: List
        """

        all_mouth_coords = list()
        if results.multi_face_landmarks:
            for face_mesh in results.multi_face_landmarks:
                mouth_coords = dict()
                for number,index in enumerate(self._mouth_indices,start = 1):
                    mouth_coords.update({number:face_mesh.landmark[index]})

                all_mouth_coords.append(mouth_coords)

        return all_mouth_coords

    def get_coordinates(self, right_eye_coords):
        """
        Metoda do uzyskania pozycji punktów w postaci trzech list dla osi x,y oraz z

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

            for key, value in face.items():
                x_list.append(value.x)
                y_list.append(1-value.y)
                z_list.append(value.z)

            x_list.append(x_list[0])
            y_list.append(y_list[0])
            z_list.append(z_list[0])

            x_list = np.array(x_list)
            y_list = np.array(y_list)
            z_list = np.array(z_list)

            x_list_all.append(x_list)
            y_list_all.append(y_list)
            z_list_all.append(z_list)

        return x_list_all, y_list_all, z_list_all