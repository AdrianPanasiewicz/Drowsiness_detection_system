import numpy as np

from .parameter_finder import ParameterFinder


class JawnFinder(ParameterFinder):
    def __init__(self):
        self.mouth_indices = np.array([[37, 84], [0, 17], [267, 314], [62,29]])
        pass

    def find_parameter(self, face_coords):
        jawn_ratios = self._find_jawn_ratio(face_coords)
        return jawn_ratios


    def _find_jawn_ratio(self,face_coords):
        """
        Metoda do obliczania szerokości otwarcia ust
        #TODO Dokumentacja
        :param face_coords: Wynik działania modelu mediapipe
        :type face_coords:
        :return:
        :rtype:
        """
        all_delta_ver_dist = np.array([])
        all_faces_jawn_ratio = list()

        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                for pair in self.mouth_indices[0:-1]:
                    delta_ver_dist = JawnFinder.calculate_distance(face_mesh, pair)
                    all_delta_ver_dist = np.append(all_delta_ver_dist, delta_ver_dist)

                hor_distance = self.calculate_distance(face_mesh, self.mouth_indices[-1])

                all_delta_ver_dist = np.array(all_delta_ver_dist)
                mean_ver_distance = np.mean(all_delta_ver_dist)

                jawn_ratio = mean_ver_distance / hor_distance
                all_faces_jawn_ratio.append(jawn_ratio)

            return all_faces_jawn_ratio
        else:
            return [0]

    @staticmethod
    def calculate_distance(face_mesh, pair):
        """


        :param face_mesh:
        :type face_mesh:
        :param pair:
        :type pair:
        :return:
        :rtype:
        """
        y2 = face_mesh.landmark[pair[0]].y
        x2 = face_mesh.landmark[pair[0]].x
        y1 = face_mesh.landmark[pair[1]].y
        x1 = face_mesh.landmark[pair[1]].x
        delta_ver_dist = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        return delta_ver_dist