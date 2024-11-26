from .parameter_finder import ParameterFinder
import numpy as np

class PerclosFinder(ParameterFinder):
    def __init__(self):
        self.left_eye_indices = [(385,380),(387,373),(263,362)]
        self.right_eye_indices = [(160,144),(158,153),(133,33)]

    def find_parameter(self, face_coords):
        """
        :param face_coords:
        :type face_coords:
        :return:
        :rtype:
        """
        left_ecr = self._find_eye_closure_ratio(face_coords, self.left_eye_indices)
        right_ecr = self._find_eye_closure_ratio(face_coords, self.right_eye_indices)

        return left_ecr, right_ecr

    @staticmethod
    def _find_eye_closure_ratio(face_coords, indices):
        """
        Metoda do obliczania szerokości otwarcia oczu

        :param face_coords: Wynik działania modelu mediapipe
        :type face_coords:
        :param indices: Wartości indeksów dla landmarków oczu
        :type indices: list
        :return:
        :rtype:
        """
        all_delta_y = list()
        all_faces_ecr = list()

        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                for pair in indices[0:-1]:
                    y2 = face_mesh.landmark[pair[0]].y
                    y1 = face_mesh.landmark[pair[1]].y

                    delta_y = abs(y2-y1)
                    all_delta_y.append(delta_y)

                x2 = face_mesh.landmark[indices[-1][0]].x
                x1 = face_mesh.landmark[indices[-1][1]].x
                x_distance = abs(x2-x1)

                all_delta_y = np.array(all_delta_y)
                mean_y_distance = np.mean(all_delta_y)

                eye_closure_ratio = mean_y_distance / x_distance
                all_faces_ecr.append(eye_closure_ratio)

            return all_faces_ecr
        else :
            return 1

    def _calculate_perclos(self, eye_closure_ratio):
        pass




