import abc

class ParameterFinder(metaclass=abc.ABCMeta):
    """
    Klasa abstrakcyjna odpowiedzialna za wyznaczanie konkretnego
    parametru na podstawie współrzędnych twarzy.
    """

    @abc.abstractmethod
    def __init__(self):
        """
        Konstruktor klasy abstrakcyjnej ParameterFinder.
        """
        pass

    @abc.abstractmethod
    def find_parameter(self, face_coords):
        """
        Metoda abstrakcyjna do zwrócenia szukanego parametru
        na podstawie wskaźników (landmarków) na twarzy.
        """
        pass
