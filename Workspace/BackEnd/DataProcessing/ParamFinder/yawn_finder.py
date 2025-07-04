import numpy as np
from .param_finder import ParamFinder


class YawnFinder(ParamFinder):
    """
    Klasa odpowiedzialna za wykrywanie ziewania na podstawie
    proporcji (stosunku pionowej wysokości ust do jej szerokości).
    """

    def __init__(self, yawn_threshold: float, is_image_mode = False):
        """
        Inicjalizuje obiekt YawnFinder, ustala indeksy ust i pamięć
        do zliczania ziewnięć, a także ustawia próg (threshold) decydujący
        o uznaniu ust za „otwarte”.

        :param yawn_threshold: Wartość progowa stosunku ust (wysokość/szerokość),
                               powyżej której uznajemy, że następuje ziewanie.
        :type yawn_threshold: float
        """
        self.mouth_indices = np.array([[37, 84], [0, 17], [267, 314], [62, 29]])
        self.yawn_counter = 0
        if is_image_mode:
            self.yawn_memory = np.zeros(1, dtype=bool)
        else:
            self.yawn_memory = np.zeros(10, dtype=bool)
        self.yawn_threshold = yawn_threshold

    def reset_memory(self):
        self.yawn_memory = np.zeros(len(self.yawn_memory), dtype=bool)

    def find_parameter(self, face_coords) -> tuple:
        """
        Główna metoda interfejsu ParamFinder. Zwraca informację,
        czy użytkownik ziewa, liczbę dotychczasowych ziewnięć oraz
        aktualny stosunek ust (MAR - Mouth Aspect Ratio).

        :param face_coords: Wynik działania MediaPipe (multi_face_landmarks),
                            przechowujący punkty siatki twarzy.
        :type face_coords: Union[...] lub podobne
        :return: Krotka (is_yawning, yawn_counter, mouth_ratio).
                 - is_yawning (bool): flaga czy ziewanie jest obecne,
                 - yawn_counter (int): liczba zliczonych ziewnięć,
                 - mouth_ratio (float): wartość stosunku ust (wysokość / szerokość).
        :rtype: tuple
        """
        yawn_ratios = self._find_yawn_ratio(face_coords)
        is_jawning = self._check_for_yawn(yawn_ratios)
        mouth_ratio = yawn_ratios[0] if face_coords.multi_face_landmarks else 0.0
        return is_jawning, self.yawn_counter, mouth_ratio

    def _find_yawn_ratio(self, face_coords) -> list:
        """
        Oblicza stosunek „wysokości” ust do ich „szerokości” (MAR – Mouth Aspect Ratio)
        dla każdej wykrytej twarzy. Jeśli nie wykryto żadnej twarzy, zwraca [0].

        :param face_coords: Obiekt zawierający multi_face_landmarks.
        :type face_coords: Union[...] lub podobne
        :return: Lista wartości MAR dla wszystkich wykrytych twarzy.
        :rtype: list
        """
        all_delta_ver_dist = np.array([])
        all_faces_yawn_ratio = []

        if face_coords.multi_face_landmarks:
            for face_mesh in face_coords.multi_face_landmarks:
                for pair in self.mouth_indices[:-1]:
                    delta_ver_dist = YawnFinder._calculate_distance(face_mesh, pair)
                    all_delta_ver_dist = np.append(all_delta_ver_dist, delta_ver_dist)

                hor_distance = self._calculate_distance(face_mesh, self.mouth_indices[-1])
                mean_ver_distance = np.mean(all_delta_ver_dist)
                jawn_ratio = mean_ver_distance / hor_distance
                all_faces_yawn_ratio.append(jawn_ratio)
            return all_faces_yawn_ratio
        else:
            return [0]

    @staticmethod
    def _calculate_distance(face_mesh, pair: list) -> float:
        """
        Zwraca odległość euklidesową między dwoma punktami (landmarkami) ust.

        :param face_mesh: Pojedyncza wykryta twarz z biblioteki MediaPipe.
        :type face_mesh: Union[...] lub podobne
        :param pair: Para indeksów punktów (landmarków),
                     na podstawie których wyliczana jest odległość.
        :type pair: list
        :return: Odległość między punktami (float).
        :rtype: float
        """
        y2 = face_mesh.landmark[pair[0]].y
        x2 = face_mesh.landmark[pair[0]].x
        y1 = face_mesh.landmark[pair[1]].y
        x1 = face_mesh.landmark[pair[1]].x
        return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    def _check_for_yawn(self, yawn_ratio: float) -> bool:
        """
        Sprawdza, czy aktualna wartość yawn_ratio przekracza próg (yawn_threshold)
        i jeśli tak, aktualizuje licznik ziewnięć. Wykorzystuje rolling memory
        (self.yawn_memory), by rozróżnić pojedyncze długie ziewnięcie od wielu
        krótkich.

        :param yawn_ratio: Lista wartości MAR dla ust (wysokość/szerokość).
        :type yawn_ratio: list
        :return: True, jeśli użytkownik aktualnie ziewa; w przeciwnym wypadku False.
        :rtype: bool
        """
        self.yawn_memory = np.roll(self.yawn_memory, -1)
        self.yawn_memory[-1] = False

        if yawn_ratio[0] >= self.yawn_threshold:
            if not any(self.yawn_memory):
                self.yawn_counter += 1
            self.yawn_memory[-1] = True
            return True

        return False
