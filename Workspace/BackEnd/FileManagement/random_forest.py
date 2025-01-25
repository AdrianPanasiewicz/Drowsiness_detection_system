import os
import pathlib
import numpy as np
import sys
import pickle
from pandas import DataFrame
from typing import Dict, Any


class RandomForest:
    """
    Klasa odpowiedzialna za wczytywanie modeli do programu. Domyślnie
    może ładować modele zdefiniowane w `default_relative_model_locations`,
    jak również dodawać nowe ścieżki do modeli.
    """



    def __init__(self, activation_certainty: float = 0.5, prediction_memory_size: int = 60) -> None:
        """
        Inicjalizuje obiekt RandomForest, ustala ścieżki do modeli
        zawartych w domyślnej liście 'default_relative_model_locations'
        i zapisuje je w słowniku 'self.model_paths'.

        :param activation_certainty: Minimalny "procent" (np. 0.5) potwierdzający senność.
        :type activation_certainty: float
        :param prediction_memory_size: Rozmiar "pamięci" (bufora) przechowującej ostatnie predykcje.
        :type prediction_memory_size: int
        """
        # Słownik przechowujący nazwy modeli i odpowiadające im ścieżki na dysku
        self.model_paths = {}

        # Wartość progowa, powyżej której uznawana jest senność
        self.activation_certainty: float = activation_certainty
        # Bufor do przechowywania ostatnich predykcji (True/False)
        self.prediction_memory: np.ndarray = np.zeros(prediction_memory_size, dtype=bool)

        base_dir = pathlib.Path(sys.argv[0]).parent  # The folder containing the .exe
        pkl_path = base_dir / "Models" / "random_forest_drowsiness_model.pkl"


        # Dodanie domyślnych ścieżek modeli do słownika self.model_paths
        try:
            with open(pkl_path, "rb") as f:
                self.random_forest = pickle.load(f)
        except Exception as e:
            print(f"Not loaded. Error: {e}")
            raise e


    def save_model_path_from_relative_path(self, relative_path: str) -> None:
        """
        Dodaje ścieżkę do modelu na podstawie ścieżki względnej względem
        głównego folderu projektu.

        :param relative_path: Względna ścieżka do pliku modelu
        :type relative_path: str
        :return: None
        """
        _working_dir = pathlib.Path(__file__).parent.parent.parent
        _model_full_path = _working_dir / relative_path
        filename = os.path.basename(_model_full_path)
        self.model_paths[filename[:-4]] = _model_full_path

    def save_model_path_from_absolute_path(self, absolute_path: str) -> None:
        """
        Dodaje ścieżkę do modelu na podstawie bezwzględnej ścieżki w systemie plików.

        :param absolute_path: Bezwzględna ścieżka do pliku modelu
        :type absolute_path: str
        :return: None
        """
        _model_full_path = pathlib.Path(absolute_path)
        filename = os.path.basename(_model_full_path)
        self.model_paths[filename[:-4]] = _model_full_path

    def load_models(self) -> Dict[str, Any]:
        """
        Wczytuje modele z zapisanych ścieżek w 'self.model_paths'
        i zwraca słownik postaci {nazwa_modelu: załadowany_model}.

        :return: Słownik z nazwą modelu jako kluczem i załadowanym modelem jako wartością.
        :rtype: dict
        """
        loaded_models = {}
        for model_name, model_path in self.model_paths.items():
            with open(model_path, 'rb') as open_file:
                self.random_forest = pickle.load(open_file)
                loaded_models[model_name] = self.random_forest
        return loaded_models

    def predict(self, data: DataFrame) -> bool:
        """
        Dokonuje predykcji senności na podstawie przekazanych danych (DataFrame).

        :param data: Zbiór cech (m.in. EAR, MAR, PERCLOS itp.) dla pojedynczej obserwacji.
        :type data: pandas.DataFrame
        :return: True w przypadku rozpoznania senności, False w przeciwnym wypadku.
        :rtype: bool
        """
        value_map = {"Drowsy": True, "Not_drowsy": False}
        prediction = value_map[self.random_forest.predict(data)[0]]
        return prediction

    def moving_mode_value_prediction(self, data: DataFrame) -> bool:
        """
        Dokonuje predykcji senności z wykorzystaniem "bufora pamięci" poprzednich predykcji.
        Jeśli w określonym oknie czasowym (prediction_memory_size) częstość wystąpień True
        przekracza próg 'activation_certainty', metoda zwróci True. W przeciwnym wypadku - False.

        :param data: Dane (cechy) do predykcji (np. z aktualnej klatki wideo).
        :type data: pandas.DataFrame
        :return: Ostateczna predykcja (True/False) uwzględniająca historyczne predykcje.
        :rtype: bool
        """
        # Predykcja z aktualnego zestawu danych
        single_prediction = self.predict(data)

        # Przesuwanie i aktualizowanie bufora predykcji
        self.prediction_memory = np.roll(self.prediction_memory, -1)
        self.prediction_memory[-1] = single_prediction

        # Zliczenie wystąpień True i False w buforze
        unique, counts = np.unique(self.prediction_memory, return_counts=True)
        predictions_votes_in_period = dict(zip(unique, counts))

        # Uzupełnienie słownika o brakujące klucze, gdyby w buforze nie było żadnej wartości True/False
        if np.True_ not in predictions_votes_in_period:
            predictions_votes_in_period[np.True_] = 0
        if np.False_ not in predictions_votes_in_period:
            predictions_votes_in_period[np.False_] = 0

        # Obliczenie "pewności" (jaki odsetek predykcji w buforze to True)
        prediction_certainty = predictions_votes_in_period[np.True_] / len(self.prediction_memory)

        # Porównanie pewności z progiem i ostateczna decyzja
        if prediction_certainty >= self.activation_certainty:
            prediction = True
        else:
            prediction = False

        return prediction
