import abc

class ParameterFinder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def find_parameter(self, face_coords):
        pass
