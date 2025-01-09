import abc
from typing import Any


class ParamFinder(metaclass=abc.ABCMeta):
    """
    Klasa abstrakcyjna odpowiedzialna za wyznaczanie konkretnego
    parametru na podstawie współrzędnych twarzy.
    """

    @abc.abstractmethod
    def __init__(self) -> None:
        """
        Konstruktor klasy abstrakcyjnej ParamFinder.
        """
        pass

    @abc.abstractmethod
    def find_parameter(self, face_coords: Any) -> Any:
        """
        Metoda abstrakcyjna do zwrócenia szukanego parametru
        na podstawie wskaźników (landmarków) na twarzy.

        :param face_coords: Obiekt zawierający współrzędne twarzy
                            (np. z biblioteki MediaPipe).
        :type face_coords: Any
        :return: Wartosc lub obiekt reprezentujący wyliczony parametr.
        :rtype: Any
        """
        pass
