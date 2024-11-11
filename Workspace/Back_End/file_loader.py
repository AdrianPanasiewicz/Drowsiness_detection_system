import os
import re

class FileLoader:
    """
    Klasa odpowiedzialna za wczytywanie modeli do programu.
    """


    def __init__(self):
        """
        Domyślny bezparametrowy konstruktor klasy FileLoader
        """
        self._working_dir = os.getcwd()
        self._model_relative_path = r'\Workspace\Resources\Models\Emotion_Model.pkl'
        self._model_full_path = os.path.join(self._working_dir, self._model_relative_path)

        # regex_model_name = r"\(\[a-Z_-]+).pkl$"
        # result = re.search(regex_model_name, self._model_full_path)

        # self.model_paths = dict()
        # self.model_paths.update({})


    @classmethod
    def from_relative_path(cls, relative_path):
        """
        Konstruktor, który tworzy klasę na podstawie relatywnej ścieżki względem folderu, w którym jest projekt

        :param relative_path: Relatywnej ścieżka względem folderu, w którym jest projekt
        :type relative_path: string
        :return: Zwraca klasę FileLoader
        :rtype: FileLoader
        """
        cls._working_dir = os.getcwd()
        cls._model_relative_path = relative_path
        cls._model_full_path = os.path.join(cls._working_dir, cls._model_relative_path)
        FileLoader.loaded_models += 1

        return cls

    @classmethod
    def from_absolute_path(cls, absolute_path):
        """
        Konstruktor, który tworzy klasę na podstawie relatywnej ścieżki względem folderu, w którym jest projekt

        :param absolute_path: Bezwzględna ścieżka, gdzie znajduje się model
        :type absolute_path: string
        :return: Zwraca klasę FileLoader
        :rtype: FileLoader
        """

        cls._model_full_path = absolute_path
        cls._working_dir = os.getcwd()
        cls._model_relative_path = os.path.relpath(absolute_path, cls._working_dir)
        FileLoader.loaded_models += 1

        return cls