import matplotlib as plt

class ParameterCalculator:
    def __init__(self):
        self._left_eye_indices = [398,384,385,386,387,388,466,263,249,390,373,374,380,381]
        self._right_eye_indices = [7,163,144,145,153,154,173,157,158,159,160,161]
        self._mouth_indices = list()
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
