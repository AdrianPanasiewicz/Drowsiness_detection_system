import numpy as np
from typing import Any, List, Tuple
from .param_finder import ParamFinder


class PerclosFinder(ParamFinder):
    """
    Klasa odpowiedzialna za obliczanie współczynnika PERCLOS (procent czasu, w którym
    powieki są zamknięte) na podstawie wartości EAR (Eye Aspect Ratio) dla lewego i prawego oka.
    """

    def __init__(self, perclos_threshold: float) -> None:
        """
        Inicjalizuje obiekt klasy PerclosFinder, ustalając m.in. pary indeksów
        niezbędne do obliczania EAR dla lewego i prawego oka oraz próg (threshold)
        stosowany do wyznaczania PERCLOS.

        :param perclos_threshold: Próg, poniżej którego oko uznawane jest za zamknięte.
        :type perclos_threshold: float
        """
        self.left_eye_indices: List[Tuple[int, int]] = [(385, 380), (387, 373), (263, 362)]
        self.right_eye_indices: List[Tuple[int, int]] = [(160, 144), (158, 153), (133, 33)]
        self.ear_per_face_memory: dict[int, dict[int, Tuple[float, float]]] = {1: {1: (0, 0)}}
        self.previous_perclos: float = 0.0
        self.perclos_threshold: float = perclos_threshold

    def find_parameter(self, face_coords: Any) -> Tuple[float, float]:
        """
        Oblicza dwa parametry związane z oczami:
        1. PERCLOS (procent czasu, gdy oczy są zamknięte),
        2. Średnią wartość EAR (Eye Aspect Ratio) obu oczu.

        :param face_coords: Współrzędne twarzy z biblioteki MediaPipe (multi_face_landmarks).
        :type face_coords: Any
        :return: Krotka (perclos, mean_ear), gdzie:
                 - perclos (float): Obliczony procent czasu zamknięcia powiek.
                 - mean_ear (float): Średnie EAR (Eye Aspect Ratio) dla obu oczu.
        :rtype: tuple[float, float]
        """
        left_ear = self._find_eye_aspect_ratio(face_coords, self.left_eye_indices)
        right_ear = self._find_eye_aspect_ratio(face_coords, self.right_eye_indices)

        if face_coords.multi_face_landmarks:
            perclos = self._calculate_perclos(left_ear[0], right_ear[0], 1)
            self.previous_perclos = perclos
            mean_ear = (left_ear[0] + right_ear[0]) / 2
        else:
            perclos = 0.0
            mean_ear = 0.0

        return perclos, mean_ear

    @staticmethod
    def _find_eye_aspect_ratio(face_coords: Any, indices: List[Tuple[int, int]]) -> List[float]:
        """
        Oblicza Eye Aspect Ratio (EAR) dla zestawu twarzy na podstawie wskazanych
        indeksów landmarków oczu. Wykorzystuje pary punktów pionowych i jedną
        parę poziomą, by wyznaczyć proporcje (delta_y/delta_x).

        :param face_coords: Obiekt multi_face_landmarks z biblioteki MediaPipe.
        :type face_coords: Any
        :param indices: Lista krotek indeksów punktów:
                        - kilka par pionowych (np. (385,380), (387,373))
                        - jedna para pozioma (np. (263,362)).
        :type indices: list[tuple[int, int]]
        :return: Lista zawierająca wartości EAR dla każdej wykrytej twarzy.
        :rtype: list[float]
        """
        all_faces_ear: List[float] = []
        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                all_delta_ver_dist = np.array([])
                # Obliczanie odległości pionowych
                for pair in indices[:-1]:
                    y2 = face_mesh.landmark[pair[0]].y
                    x2 = face_mesh.landmark[pair[0]].x
                    y1 = face_mesh.landmark[pair[1]].y
                    x1 = face_mesh.landmark[pair[1]].x

                    delta_ver_dist = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
                    all_delta_ver_dist = np.append(all_delta_ver_dist, delta_ver_dist)

                # Obliczanie odległości poziomej
                y2 = face_mesh.landmark[indices[-1][0]].y
                x2 = face_mesh.landmark[indices[-1][0]].x
                y1 = face_mesh.landmark[indices[-1][1]].y
                x1 = face_mesh.landmark[indices[-1][1]].x
                hor_distance = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5

                # EAR = (średnia z pionowych) / pozioma
                mean_ver_distance = float(np.mean(all_delta_ver_dist))
                eye_aspect_ratio = mean_ver_distance / hor_distance
                # Zakładamy, że wartość EAR nie może przekroczyć 1
                eye_aspect_ratio = np.clip(eye_aspect_ratio, 0, 1)
                all_faces_ear.append(float(eye_aspect_ratio))

        return all_faces_ear

    def _calculate_perclos(
        self,
        left_eye_aspect_ratio: float,
        right_eye_aspect_ratio: float,
        memory_key: int
    ) -> float:
        """
        Oblicza współczynnik PERCLOS (procent czasu, gdy oczy są zamknięte), bazując na historii
        odczytów EAR dla danej twarzy (zidentyfikowanej przez 'memory_key'). Dla każdej klatki
        obliczana jest średnia EAR z obydwu oczu i porównywana z progiem (self.perclos_threshold).

        :param left_eye_aspect_ratio: EAR lewego oka.
        :type left_eye_aspect_ratio: float
        :param right_eye_aspect_ratio: EAR prawego oka.
        :type right_eye_aspect_ratio: float
        :param memory_key: Unikalny identyfikator twarzy, dla której liczymy PERCLOS.
        :type memory_key: int
        :return: Procent klatek, w których oczy były uznane za zamknięte.
        :rtype: float
        """
        period = 900  # liczba klatek uwzględnianych w pamięci (np. 900 ~ 30s przy 30 FPS)

        # Usunięcie najstarszego wpisu, jeśli osiągnięto limit 'period'
        if len(self.ear_per_face_memory[memory_key]) >= period:
            oldest_frame = min(self.ear_per_face_memory[memory_key].keys())
            del self.ear_per_face_memory[memory_key][oldest_frame]

        # Dodanie nowego wpisu dla aktualnej klatki
        latest_frame = max(self.ear_per_face_memory[memory_key].keys())
        ecr_ratios: Tuple[float, float] = (left_eye_aspect_ratio, right_eye_aspect_ratio)
        self.ear_per_face_memory[memory_key].update({latest_frame + 1: ecr_ratios})

        # Zliczanie klatek, w których średnia EAR < próg (oczy zamknięte)
        closed_count = 0
        for _, pair in self.ear_per_face_memory[memory_key].items():
            mean_from_pair = (pair[0] + pair[1]) / 2
            if mean_from_pair < self.perclos_threshold:
                closed_count += 1

        # PERCLOS to odsetek (closed_count / period)
        perclos = closed_count / period
        return perclos
