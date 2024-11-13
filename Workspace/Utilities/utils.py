import time

class Utils:

    _past_tick = 0

    @classmethod
    def calculate_fps(cls):
        """
        Metoda do obliczania wartości klatek na sekundę.

        :return: Ilość klatek na sekundę.
        :rtype: Float
        """

        current_tick = time.time()
        fps = 1/(current_tick - cls._past_tick)
        cls._past_tick = current_tick
        
        return fps


