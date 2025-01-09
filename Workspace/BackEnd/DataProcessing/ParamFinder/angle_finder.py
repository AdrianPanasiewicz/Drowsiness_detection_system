import math
import numpy as np
from typing import Tuple, Any
from .param_finder import ParamFinder


class AngleFinder(ParamFinder):
    """
    Klasa odpowiedzialna za wyznaczanie kąta pochylenia twarzy (roll i pitch)
    na podstawie wybranych wskaźników (landmarków) twarzy pochodzących z MediaPipe.
    """

    def __init__(self, roll_memory_size = 15, pitch_memory_size = 15) -> None:
        """
        Inicjalizuje obiekt AngleFinder poprzez zdefiniowanie zbioru par indeksów,
        na których będzie przeprowadzana analiza kąta pochylenia twarzy.
        """
        self.face_oval_indices: np.ndarray = np.array([[109, 148], [10, 152], [338, 377]])
        self.roll_memory = np.array(np.zeros(roll_memory_size))
        self.pitch_memory = np.array(np.zeros(pitch_memory_size))

    def find_parameter(self, face_coords: Any) -> Tuple[float, float]:
        """
        Główna metoda interfejsu ParamFinder. Zwraca kąt przechylenia (roll)
        i kąt pochylenia (pitch) twarzy dla przekazanego zestawu współrzędnych.

        :param face_coords: Wynik działania biblioteki MediaPipe, zawierający
                            obiekt multi_face_landmarks z wykrytymi punktami twarzy.
                            Zwykle: face_mesh_results = mediapipe.python.solutions.face_mesh.FaceMesh.process(image)
        :type face_coords: Any (np. obiekt z atrybutem multi_face_landmarks)
        :return: Krotka (roll, pitch) wyrażona w stopniach.
        :rtype: Tuple[float, float]
        """
        roll, pitch = self._find_face_angle(face_coords)
        return roll, pitch

    def _find_face_angle(self, face_coords: Any) -> Tuple[float, float]:
        """
        Oblicza średnie wartości kąta przechylenia (roll) i pochylenia (pitch) twarzy,
        bazując na parach punktów w `self.face_oval_indices`.

        :param face_coords: Obiekt zawierający listę wykrytych twarzy (multi_face_landmarks).
        :type face_coords: Any
        :return: Krotka (roll, pitch) w stopniach (lub (0, 0), jeśli nie wykryto żadnej twarzy).
        :rtype: Tuple[float, float]
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

            roll = float(np.mean(roll_estimates))
            pitch = float(np.mean(pitch_estimates))
            self.roll_memory = np.roll(self.roll_memory, -1)
            self.pitch_memory = np.roll(self.pitch_memory, -1)
            self.roll_memory[-1] = roll
            self.pitch_memory[-1] = pitch

            mean_roll = float(np.mean(self.roll_memory))
            mean_pitch = float(np.mean(self.pitch_memory))

            return mean_roll, mean_pitch
        else:
            return 0.0, 0.0

    @staticmethod
    def _calculate_euler_angles(face_mesh: Any, pair: np.ndarray) -> Tuple[float, float]:
        """
        Na podstawie dwóch wskaźników (landmarków) twarzy oblicza kąt przechylenia (roll)
        oraz kąt pochylenia (pitch) w stopniach.

        :param face_mesh: Pojedynczy obiekt reprezentujący twarz (np. jeden element z multi_face_landmarks).
        :type face_mesh: Any
        :param pair: Tablica zawierająca dwa indeksy landmarków (np. [10, 152]).
        :type pair: np.ndarray
        :return: Krotka (roll, pitch) w stopniach.
        :rtype: Tuple[float, float]
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

        # Obliczenia kątów w stopniach (atan2 zwraca wynik w radianach).
        roll = math.atan2(abs(delta_z), delta_x) * 180.0 / math.pi - 90.0
        pitch = math.atan2(delta_z, abs(delta_y)) * 180.0 / math.pi

        return roll, pitch
