import os
import pathlib
import fastai.vision.all


class ModelLoader:
    """
    Klasa odpowiedzialna za wczytywanie modeli do programu. Domyślnie
    może ładować modele zdefiniowane w `default_relative_model_locations`,
    jak również dodawać nowe ścieżki do modeli.
    """

    default_relative_model_locations = [r'Resources\Models\Emotion_model.pkl']

    def __init__(self):
        """
        Inicjalizuje obiekt ModelLoader, ustala ścieżki do modeli
        zawartych w domyślnej liście 'default_relative_model_locations'
        i zapisuje je w słowniku 'self.model_paths'.
        """
        self.model_paths = {}
        for path in self.default_relative_model_locations:
            _working_dir = pathlib.Path(__file__).parent.parent.parent
            _model_full_path = _working_dir / pathlib.Path(path)
            filename = os.path.basename(_model_full_path)
            self.model_paths[filename[:-4]] = _model_full_path

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
            loaded_models[model_name] = fastai.vision.all.load_learner(model_path)
        return loaded_models
