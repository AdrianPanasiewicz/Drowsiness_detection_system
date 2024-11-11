import os
import pathlib

class FileLoader:
    """
    Klasa odpowiedzialna za wczytywanie modeli do programu.
    """
    default_relative_model_locations = [r'Workspace\Resources\Models\Emotion_model.pkl', r'Workspace\Resources\Models\Drowsiness_model.pkl']

    def __init__(self):
        """
        Domyślny bezparametrowy konstruktor klasy FileLoader
        """
        self.model_paths =  dict()

        for path in self.default_relative_model_locations:
            self._working_dir = pathlib.Path(__file__).parent.parent.parent
            self._model_relative_path = pathlib.Path(path)
            self._model_full_path = self._working_dir / self._model_relative_path

            filename =  os.path.basename(self._model_full_path)
            self.model_paths.update({filename[:-4] : self._model_full_path})


    def load_model_from_relative_path(self, relative_path):
        """
        Metoda, która pozwala na załadowanie modelu ze względnej ścieżki

        :param relative_path: Relatywnej ścieżka względem folderu, w którym jest projekt
        :type relative_path: string
        :return: Zwraca klasę FileLoader
        :rtype: FileLoader
        """
        # cls._working_dir = os.getcwd()
        # cls._model_relative_path = relative_path
        # cls._model_full_path = os.path.join(cls._working_dir, cls._model_relative_path)
        # FileLoader.loaded_models += 1
        #
        # return cls

    def load_model_from_absolute_path(self, absolute_path):
        """
        Konstruktor, który tworzy klasę na podstawie relatywnej ścieżki względem folderu, w którym jest projekt

        :param absolute_path: Bezwzględna ścieżka, gdzie znajduje się model
        :type absolute_path: string
        :return: Zwraca klasę FileLoader
        :rtype: FileLoader
        """

        # cls._model_full_path = absolute_path
        # cls._working_dir = os.getcwd()
        # cls._model_relative_path = os.path.relpath(absolute_path, cls._working_dir)
        # FileLoader.loaded_models += 1
        #
        # return cls

