import numpy as np
import time


from .parameter_finder import ParameterFinder


class SaccadeVelocityFinder(ParameterFinder):
    def __init__(self):
        self.right_iris_indices = np.array([469,470,471,472])
        self.left_iris_indices = np.array([474,475,476,477])
        self.iris_previous_state = np.zeros((2, 4, 4))
        self.landmark_previous_state = np.zeros((2, 2, 3))

    def find_parameter(self, face_coords):
        """

        :param face_coords:
        :type face_coords:
        :return:
        :rtype:
        """
        right_saccade_velocity = self._find_saccade_velocity(face_coords,self.right_iris_indices, right_flag = True)
        left_saccade_velocity = self._find_saccade_velocity(face_coords, self.left_iris_indices, left_flag = True)
        mean_saccade_velocity = (right_saccade_velocity + left_saccade_velocity)/2

        return mean_saccade_velocity

    def _find_saccade_velocity(self, face_coords, indices, left_flag = False, right_flag = False):
        """

        #TODO Dokumentacja
        :param face_coords: Wynik działania modelu mediapipe
        :type face_coords:
        :return:
        :rtype:
        """
        all_landmark_saccade_velocity = np.array([])

        if left_flag:
            iris_selection = 1
            adjust_to_landmarks = [33,133]
        elif right_flag:
            iris_selection = 0
            adjust_to_landmarks = [362,263]
        else:
            iris_selection = 0
            adjust_to_landmarks = [362,263]

        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                for index, landmark in enumerate(indices,start=0):
                    landmark_velocity = self._calculate_velocity(face_mesh, landmark, index, iris_selection, adjust_to_landmarks)
                    all_landmark_saccade_velocity = np.append(all_landmark_saccade_velocity, landmark_velocity)

            saccade_velocity = np.mean(all_landmark_saccade_velocity)

            return saccade_velocity
        else:
            return 0

    def _calculate_velocity(self,face_mesh, landmark_ind, index, iris_selection, adjust_to_landmarks):
        """


        :param face_mesh:
        :type face_mesh:
        :param landmark_ind:
        :type landmark_ind:
        :return:
        :rtype:
        """
        prev_state = self.iris_previous_state[iris_selection][index]
        x2 = prev_state[0]
        y2 = prev_state[1]
        z2 = prev_state[2]
        tick2 = prev_state[3]

        x1 = face_mesh.landmark[landmark_ind].x
        y1 = face_mesh.landmark[landmark_ind].y
        z1 = face_mesh.landmark[landmark_ind].z
        tick1 = time.time()

        delta_x = x1 - x2
        delta_y = y1 - y2
        delta_z = z1 - z2
        delta_t = tick1 - tick2

        distance = (delta_x**2 + delta_y**2 + delta_z**2)**0.5
        adjust_by_distance = self.adjust_speed_to_landmarks(adjust_to_landmarks,iris_selection, face_mesh)
        adjusted_distance = distance - adjust_by_distance

        saccade_velocity = adjusted_distance/delta_t

        self.iris_previous_state[iris_selection][index][0] = x1
        self.iris_previous_state[iris_selection][index][1] = y1
        self.iris_previous_state[iris_selection][index][2] = z1
        self.iris_previous_state[iris_selection][index][3] = tick1

        return saccade_velocity

    def adjust_speed_to_landmarks(self, adjust_to_landmarks, iris_selection, face_mesh):
        """

        :param adjust_to_landmarks:
        :type adjust_to_landmarks:
        :param iris_selection:
        :type iris_selection:
        :param face_mesh:
        :type face_mesh:
        :return:
        :rtype:
        """
        distances = np.zeros(2)
        for index, landmark_ind in enumerate(adjust_to_landmarks,start=0):
            prev_state = self.landmark_previous_state[iris_selection][index]
            x2 = prev_state[0]
            y2 = prev_state[1]
            z2 = prev_state[2]

            x1 = face_mesh.landmark[landmark_ind].x
            y1 = face_mesh.landmark[landmark_ind].y
            z1 = face_mesh.landmark[landmark_ind].z

            delta_x = x1 - x2
            delta_y = y1 - y2
            delta_z = z1 - z2

            distance = (delta_x**2 + delta_y**2 + delta_z**2)**0.5

            distances[index] = distance

        adjust_by_distance = np.mean(distances)

        return adjust_by_distance