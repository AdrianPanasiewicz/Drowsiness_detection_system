import time
import pathlib

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

    @classmethod
    def fix_pathlib(self):
        # A work-around for the error that PosixPath cannot be instantiated on your system
        # More about this issue on https://github.com/ultralytics/yolov5/issues/10240
        temp = pathlib.PosixPath
        fixed_path = pathlib.WindowsPath
        return fixed_path
