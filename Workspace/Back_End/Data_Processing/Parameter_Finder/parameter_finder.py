import abc

class ParameterFinder(metaclass=abc.ABCMeta):
    """Klasa odpowiedzialna za znalezienie konkretnego parametru"""
    @abc.abstractmethod
    def __init__(self):
        """Konstruktor klasy"""
        pass

    @abc.abstractmethod
    def find_parameter(self, face_coords):
        """Metoda do zwrócenia szukanego parametru na podstawie współrzędnych wskaźników na twarzy."""
        pass
