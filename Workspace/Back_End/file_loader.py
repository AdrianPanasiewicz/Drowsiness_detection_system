import os
import pathlib

class FileLoader:
    """
    Klasa odpowiedzialna za wczytywanie modeli do programu.
    """
    # r'Workspace\Resources\Models\Drowsiness_model.pkl'
    default_relative_model_locations = [r'Workspace\Resources\Models\Emotion_model.pkl']

    def __init__(self):
        """
        Domyślny bezparametrowy konstruktor klasy FileLoader
        """
        self.model_paths =  dict()

        for path in self.default_relative_model_locations:
            _working_dir = pathlib.Path(__file__).parent.parent.parent
            _model_relative_path = pathlib.Path(path)
            _model_full_path = _working_dir / _model_relative_path

            filename =  os.path.basename(_model_full_path)
            self.model_paths.update({filename[:-4] : _model_full_path})


    def load_model_from_relative_path(self, relative_path: str):
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

    def load_model_from_absolute_path(self, absolute_path):
        """
        Metoda, która pozwala na załadowanie modelu z bezwzględnej ścieżki

        :param absolute_path: Bezwzględna ścieżka, gdzie znajduje się model
        :type absolute_path: string
        """

        _model_full_path = pathlib.Path(absolute_path)

        filename = os.path.basename(_model_full_path)
        self.model_paths.update({filename[:-4] : _model_full_path})