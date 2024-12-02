import numpy as np

from .parameter_finder import ParameterFinder


class JawnFinder(ParameterFinder):
    def __init__(self, yawn_threshold):
        self.mouth_indices = np.array([[37, 84], [0, 17], [267, 314], [62,29]])
        self.yawn_counter = 0
        self.yawn_memory = np.array(np.zeros(10),dtype='bool')
        self.yawn_threshold = yawn_threshold


    def find_parameter(self, face_coords):
        jawn_ratios = self._find_jawn_ratio(face_coords)
        is_jawning = self.detect_yawn(jawn_ratios)
        return is_jawning, self.yawn_counter


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

    def detect_yawn(self, yawn_ratio):
        """

        :return:
        :rtype:
        """
        self.yawn_memory = np.roll(self.yawn_memory, -1)
        self.yawn_memory[-1] = False
        if yawn_ratio[0] >= self.yawn_threshold:
            if not any(self.yawn_memory):
                self.yawn_counter += 1

            self.yawn_memory[-1] = True
            return True

        return False
