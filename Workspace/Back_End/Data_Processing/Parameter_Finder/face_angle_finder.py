import math
import numpy as np
from .parameter_finder import ParameterFinder


class FaceAngleFinder(ParameterFinder):
    """
    Klasa odpowiedzialna za wyznaczanie kąta pochylenia twarzy (roll i pitch)
    na podstawie wybranych wskaźników (landmarków) twarzy pochodzących z MediaPipe.
    """

    def __init__(self):
        """
        Inicjalizuje obiekt FaceAngleFinder poprzez zdefiniowanie zbioru par indeksów,
        na których będzie przeprowadzana analiza kąta pochylenia twarzy.
        """
        self.face_oval_indices = np.array([[109, 148], [10, 152], [338, 377]])

    def find_parameter(self, face_coords) -> tuple:
        """
        Główna metoda interfejsu ParameterFinder. Zwraca kąt przechylenia (roll)
        i kąt pochylenia (pitch) twarzy dla przekazanego zestawu współrzędnych.

        :param face_coords: Wynik działania biblioteki MediaPipe, zawierający
                            obiekt multi_face_landmarks z wykrytymi punktami twarzy.
        :type face_coords: Union[mediapipe.python.solutions.face_mesh.FaceMesh, ...] lub podobny
        :return: Krotka (roll, pitch) wyrażona w stopniach.
        :rtype: tuple
        """
        roll, pitch = self._find_face_angle(face_coords)
        return roll, pitch

    def _find_face_angle(self, face_coords) -> tuple:
        """
        Oblicza średnie wartości kąta przechylenia (roll) i pochylenia (pitch) twarzy,
        bazując na parach punktów w `self.face_oval_indices`.

        :param face_coords: Obiekt zawierający listę wykrytych twarzy (face_mesh).
        :type face_coords: Union[mediapipe.python.solutions.face_mesh.FaceMesh, ...] lub podobny
        :return: Krotka (roll, pitch) w stopniach (lub (0, 0), jeśli nie wykryto żadnej twarzy).
        :rtype: tuple
        """
        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                roll_estimates = []
                pitch_estimates = []

                # Pobieranie kątów dla każdej pary wskaźników zdefiniowanej w face_oval_indices
                for pair in self.face_oval_indices[0:-1]:
                    roll_val, pitch_val = self._calculate_euler_angles(face_mesh, pair)
                    roll_estimates.append(roll_val)
                    pitch_estimates.append(pitch_val)

            roll_mean = np.mean(roll_estimates)
            pitch_mean = np.mean(pitch_estimates)
            return roll_mean, pitch_mean
        else:
            return 0, 0

    @staticmethod
    def _calculate_euler_angles(face_mesh, pair: np.array) -> tuple:
        """
        Na podstawie dwóch wskaźników (landmarków) twarzy oblicza kąt przechylenia (roll)
        oraz kąt pochylenia (pitch) w stopniach.

        :param face_mesh: Pojedynczy obiekt reprezentujący twarz (z multi_face_landmarks).
        :type face_mesh: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList lub podobny
        :param pair: Tablica zawierająca dwa indeksy landmarków.
        :type pair: np.array
        :return: Krotka (roll, pitch) w stopniach.
        :rtype: tuple
        """
        x2 = face_mesh.landmark[pair[0]].x
        y2 = face_mesh.landmark[pair[0]].y
        z2 = face_mesh.landmark[pair[0]].z

        x1 = face_mesh.landmark[pair[1]].x
        y1 = face_mesh.landmark[pair[1]].y
        z1 = face_mesh.landmark[pair[1]].z

        delta_x = x2 - x1
        delta_y = y2 - y1
        delta_z = z2 - z1

        # Obliczenia kątów w stopniach (atan2 zwraca wynik w radianach)
        roll = math.atan2(abs(delta_z), delta_x) * 180.0 / math.pi - 90.0
        pitch = math.atan2(delta_z, abs(delta_y)) * 180.0 / math.pi

        return roll, pitch
