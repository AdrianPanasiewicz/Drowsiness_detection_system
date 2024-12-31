import numpy as np

from .parameter_finder import ParameterFinder


class PerclosFinder(ParameterFinder):
    """Klasa do obliczania parametru PERCLOS"""

    def __init__(self, perclos_threshold):
        """Konstruktor klasy PerclosFinder"""

        self.left_eye_indices = [(385, 380), (387, 373), (263, 362)]
        self.right_eye_indices = [(160, 144), (158, 153), (133, 33)]
        self.ear_per_face_memory = {1: {1: (0, 0)}}
        self.previous_perclos = 0
        self.perclos_threshold = perclos_threshold

    def find_parameter(self, face_coords) -> tuple:
        """
        Określa specyficzne parametry, w tym PERCLOS oraz średnie EAR (Eye Aspect Ratio),
        na podstawie podanych współrzędnych wykrytych punktów charakterystycznych twarzy.
        PERCLOS jest obliczany przy użyciu wcześniej zapisanych metryk, jeśli nie zostaną
        wykryte punkty charakterystyczne twarzy. Metoda zwraca dwie wyliczone metryki.

        :param face_coords: Współrzędne punktów charakterystycznych twarzy wymagane do
            obliczenia EAR i PERCLOS.
        :type face_coords: Any
        :return: Krotka zawierająca dwa elementy:
            1. `perclos` (float): Obliczona wartość PERCLOS reprezentująca częstotliwość mrugania.
            2. `mean_ear` (float): Średni Eye Aspect Ratio dla obu oczu.
        :rtype: tuple[float, float]
        """

        left_ear = self._find_eye_aspect_ratio(face_coords, self.left_eye_indices)
        right_ear = self._find_eye_aspect_ratio(face_coords, self.right_eye_indices)

        if face_coords.multi_face_landmarks:
            perclos = self._calculate_perclos(left_ear[0], right_ear[0], 1)
            self.previous_perclos = perclos
            mean_ear = (left_ear[0] + right_ear[0]) / 2
        else:
            perclos = self.previous_perclos
            mean_ear = 0

        return perclos, mean_ear

    @staticmethod
    def _find_eye_aspect_ratio(face_coords, indices: list) -> list:
        """
        Oblicza współczynnik Eye Aspect Ratio (EAR) dla podanego zestawu współrzędnych twarzy
        i indeksów punktów charakterystycznych. Funkcja wykorzystuje określone punkty charakterystyczne
        twarzy (indeksy) do obliczenia pionowych odległości oraz odległości poziomej,
        a następnie wyprowadza wartość EAR dla wykrytych twarzy. Wartości EAR są ograniczone
        do zakresu od 0 do 1 i odpowiadają każdej twarzy wykrytej w dostarczonych danych
        o współrzędnych twarzy.

        :param face_coords: Punkty charakterystyczne twarzy oraz dane dotyczące wielu twarzy.
            Zawiera informacje o punktach siatki twarzy wykorzystywane do obliczania
            współczynników proporcji.
        :type face_coords: `mediapipe.python.solutions.face_mesh`.
        :param indices: Lista par indeksów punktów charakterystycznych dla wyrównania pionowego,
            a także pojedynczej pary indeksów dla wyrównania poziomego wykorzystywanych
            do obliczania odległości i proporcji.
        :type indices: list
        :return: Lista współczynników Eye Aspect Ratio (EAR) dla wykrytych twarzy
            w dostarczonych danych o współrzędnych twarzy.
        :rtype: list
        """

        all_delta_ver_dist = np.array([])
        all_faces_ear = list()

        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                for pair in indices[0:-1]:
                    y2 = face_mesh.landmark[pair[0]].y
                    x2 = face_mesh.landmark[pair[0]].x

                    y1 = face_mesh.landmark[pair[1]].y
                    x1 = face_mesh.landmark[pair[1]].x

                    delta_ver_dist = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
                    all_delta_ver_dist = np.append(all_delta_ver_dist, delta_ver_dist)

                y2 = face_mesh.landmark[indices[-1][0]].y
                x2 = face_mesh.landmark[indices[-1][0]].x

                y1 = face_mesh.landmark[indices[-1][1]].y
                x1 = face_mesh.landmark[indices[-1][1]].x
                hor_distance = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5

                all_delta_ver_dist = np.array(all_delta_ver_dist)
                mean_ver_distance = np.mean(all_delta_ver_dist)

                eye_aspect_ratio = mean_ver_distance / hor_distance
                eye_aspect_ratio = np.clip(eye_aspect_ratio, 0, 1)
                all_faces_ear.append(eye_aspect_ratio)

        return all_faces_ear

    def _calculate_perclos(self, left_eye_aspect_ratio: float, right_eye_aspect_ratio: float,
                           memory_key: int) -> float:
        """
        Oblicza procent zamknięcia powiek (PERCLOS) dla konkretnej twarzy zidentyfikowanej
        przez `memory_key`. Metoda działa, wykorzystując zapisane wartości współczynników
        EAR (Eye Aspect Ratio) dla lewego i prawego oka, przechowując dane z ustalonej liczby
        ostatnich klatek.

        Metoda zapewnia, że dane dla każdej twarzy nie przekraczają określonego okresu,
        usuwając najstarsze dane przy dodawaniu nowych wartości. Oblicza wartość PERCLOS,
        która jest proporcją klatek, w których średnia wartość współczynników EAR dla lewego
        i prawego oka jest niższa od zdefiniowanego progu.

        :param left_eye_aspect_ratio: Współczynnik EAR dla lewego oka
        :type left_eye_aspect_ratio: float
        :param right_eye_aspect_ratio: Współczynnik EAR dla prawego oka
        :type right_eye_aspect_ratio: float
        :param memory_key: Klucz identyfikujący konkretną twarz, dla której obliczany jest PERCLOS
        :type memory_key: int
        :return: Obliczona wartość PERCLOS
        :rtype: float
        """
        period = 900  # frames
        # Usuń najstarszą klatkę i dodaj obecną, jeśli okres jest dłuższy niż 10s
        if len(self.ear_per_face_memory[memory_key]) >= period:  # TODO Okres*FPS
            oldest_frame = min(self.ear_per_face_memory[memory_key].keys())
            del self.ear_per_face_memory[memory_key][oldest_frame]

        latest_frame = max(self.ear_per_face_memory[memory_key].keys())
        ecr_ratios = (left_eye_aspect_ratio, right_eye_aspect_ratio)
        self.ear_per_face_memory[memory_key].update({latest_frame + 1: ecr_ratios})

        perclos = 0
        for _, pair in self.ear_per_face_memory[memory_key].items():
            mean_from_pair = pair[0] + pair[1]
            if mean_from_pair < self.perclos_threshold:
                perclos += 1

        perclos = perclos / period

        return perclos
