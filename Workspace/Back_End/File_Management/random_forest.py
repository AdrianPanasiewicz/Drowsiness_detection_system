import os
import pathlib
import numpy as np
import pickle
from pandas import DataFrame


class RandomForest:
    """
    Klasa odpowiedzialna za wczytywanie modeli do programu. Domyślnie
    może ładować modele zdefiniowane w `default_relative_model_locations`,
    jak również dodawać nowe ścieżki do modeli.
    """

    default_relative_model_locations = [r'Models\random_forest_drowsiness_model.pkl']

    def __init__(self, activation_certainty = 0.5, prediction_memory_size = 20):
        """
        Inicjalizuje obiekt RandomForest, ustala ścieżki do modeli
        zawartych w domyślnej liście 'default_relative_model_locations'
        i zapisuje je w słowniku 'self.model_paths'.
        """
        self.model_paths = {}
        self.activation_certainty = activation_certainty
        self.prediction_memory = np.zeros(prediction_memory_size, dtype=bool)


        for path in self.default_relative_model_locations:
            _working_dir = pathlib.Path(__file__).parent.parent.parent
            _model_full_path = _working_dir / pathlib.Path(path)
            filename = os.path.basename(_model_full_path)
            self.model_paths[filename[:-4]] = _model_full_path

        self.random_forest = self.load_models()["random_forest_drowsiness_model"]

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

    def load_models(self) -> dict:
        """
        Wczytuje modele z zapisanych ścieżek w 'self.model_paths'
        i zwraca słownik postaci {nazwa_modelu: załadowany_model}.

        :return: Słownik z nazwą modelu jako kluczem i załadowanym modelem jako wartością
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

        :param data:
        :type data:
        :return:
        :rtype:
        """
        value_map = {"Drowsy":True, "Not_drowsy":False}
        prediction = value_map[self.random_forest.predict(data)[0]]
        return prediction

    def moving_mode_value_prediction(self, data: DataFrame) -> bool:
        """

        :return:
        :rtype:
        """
        single_prediction = self.predict(data)
        self.prediction_memory = np.roll(self.prediction_memory, -1)
        self.prediction_memory[-1] = single_prediction
        unique, counts = np.unique(self.prediction_memory, return_counts=True)
        predictions_votes_in_period = dict(zip(unique, counts))
        if np.True_ not in predictions_votes_in_period:
            predictions_votes_in_period[np.True_] = 0
        if np.False_ not in predictions_votes_in_period:
            predictions_votes_in_period[np.False_] = 0
        prediction_certainty = predictions_votes_in_period[np.True_] / len(self.prediction_memory)
        if prediction_certainty >= self.activation_certainty:
            prediction = True
        else:
            prediction = False

        return prediction
