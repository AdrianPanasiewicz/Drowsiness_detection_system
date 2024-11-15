import matplotlib as plt

class ParameterCalculator:
    def __init__(self):
        self._left_eye_indices = list()
        self._right_eye_indices = list()
        self._mouth_indices = list()
        self._iris_indices = list()

    def find_left_eye(self, results) -> dict:
        """
        Metoda do zawracania współrzędnych punktów orientacyjnych lewego oka
        
        :param results: Wynik działania funkcji process od mediapipe
        :type results:
        :return: biblioteka ze współrzędnymi punktów orientacyjnych lewego oka
        :rtype: dict
        """

        left_eye_coords = dict()
        if results.multi_face_landmarks:
            for number,face_mesh in enumerate(results.multi_face_landmarks,start=1):
                for index in self._left_eye_indices:
                    left_eye_coords.update({number:face_mesh.landmark[index]})
