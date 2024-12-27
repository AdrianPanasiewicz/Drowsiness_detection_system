import os
import pathlib

import fastai.vision.all


class ModelLoader:
    """
    Klasa odpowiedzialna za wczytywanie modeli do programu.
    """
    # r'Workspace\Resources\Models\Drowsiness_model.pkl'
    default_relative_model_locations = [r'Resources\Models\Emotion_model.pkl']

    def __init__(self):
        """
        Domyślny bezparametrowy konstruktor klasy ModelLoader
        """
        self.model_paths =  dict()

        for path in self.default_relative_model_locations:
            _working_dir = pathlib.Path(__file__).parent.parent.parent
            _model_relative_path = pathlib.Path(path)
            _model_full_path = _working_dir / _model_relative_path

            filename =  os.path.basename(_model_full_path)
            self.model_paths.update({filename[:-4] : _model_full_path})


    def save_model_path_from_relative_path(self, relative_path: str):
        """
        Metoda, która pozwala na załadowanie modelu ze względnej ścieżki

        :param relative_path: Relatywnej ścieżka względem folderu, w którym jest projekt, więc dla przykładu, gdy należy
        wczytać model z folderu Models, należy zamieścić następującą ścieżkę
        r'Workspace\Resources\Models\Emotion_model.pkl'
        :type relative_path: str
        """
        _working_dir = pathlib.Path(__file__).parent.parent.parent
        _model_relative_path = relative_path
        _model_full_path = _working_dir / _model_relative_path

        filename = os.path.basename(_model_full_path)
        self.model_paths.update({filename[:-4] : _model_full_path})

    def save_model_path_from_absolute_path(self, absolute_path):
        """
        Metoda, która pozwala na załadowanie modelu z bezwzględnej ścieżki

        :param absolute_path: Bezwzględna ścieżka, gdzie znajduje się model
        :type absolute_path: string
        """

        _model_full_path = pathlib.Path(absolute_path)

        filename = os.path.basename(_model_full_path)
        self.model_paths.update({filename[:-4] : _model_full_path})

    def load_models(self) -> dict:
        """
        Wczytanie parametrów modeli

        :return: Parametry modeli
        :rtype: tuple
        """
        loaded_models = dict()

        for model_name,model_path in self.model_paths.items():
            loaded_models.update({model_name : fastai.vision.all.load_learner(model_path)})

        return loaded_models
