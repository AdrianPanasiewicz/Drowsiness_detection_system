import numpy as np
import math

from .parameter_finder import ParameterFinder

class FaceAngleFinder(ParameterFinder):
    def __init__(self):
        self.face_oval_indices = np.array([[109,148],[10,152],[338,377]])

    def find_parameter(self, face_coords):
        face_angle = self._find_face_angle(face_coords)
        return face_angle

    def _find_face_angle(self, face_coords):
        """

        #TODO Dokumentacja
        :param face_coords: Wynik działania modelu mediapipe
        :type face_coords:
        :return:
        :rtype:
        """
        all_faces_face_angle = np.array([])

        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                for pair in self.face_oval_indices[0:-1]:
                    face_angle = FaceAngleFinder.calculate_angle(face_mesh, pair)
                    all_faces_face_angle = np.append(all_faces_face_angle, face_angle)

            face_angle = np.mean(all_faces_face_angle)

            return face_angle
        else:
            return 0

    @staticmethod
    def calculate_angle(face_mesh, pair):
        """


        :param face_mesh:
        :type face_mesh:
        :param pair:
        :type pair:
        :return:
        :rtype:
        """
        x2 = face_mesh.landmark[pair[0]].x
        y2 = face_mesh.landmark[pair[0]].y
        z2 = face_mesh.landmark[pair[0]].z

        x1 = face_mesh.landmark[pair[1]].x
        y1 = face_mesh.landmark[pair[1]].y
        z1 = face_mesh.landmark[pair[1]].z

        delta_x = abs(x2 - x1)
        delta_y = abs(y2 - y1)
        delta_z = abs(z2 - z1)

        denominator = (delta_x * delta_x + delta_y + delta_y)**(1/2)
        face_angle = math.atan2(delta_z,denominator)*180/math.pi

        return face_angle