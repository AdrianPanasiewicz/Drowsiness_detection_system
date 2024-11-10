from fastai.vision.all import *
import cv2

class ImageProcessor:
    def __init__(self):
        self._original_image = None
        self._processed_image = None

    def _load_image(self, image: Tensor):
        """
        Metoda do zapisywania obrazu TensorImage do klasy

        :param image: An image from TensorImage class
        """
        self._original_image = image

    def _crop_image(self, crop_width: int, crop_height: int):
        """
        Metoda do wykonania operacji crop na obrazie zapisanym w klasie

        :param crop_width: Szerokość, z którą ma być wykonana operacja crop
        :param crop_height: Wysokość, z którą ma być wykonana operacja crop
        """
        # Uzyskanie wymiarow obrazu
        x_size = self._original_image.shape[0]
        y_size = self._original_image.shape[1]

        # Obliczenie wspolrzednych do wykonania operacji crop o wymiarach crop_width i crop_height
        x_start_crop = (x_size - crop_width)//2
        x_end_crop = (x_size + crop_width)//2

        y_start_crop = (y_size - crop_height)//2
        y_end_crop = (y_size + crop_height)//2


        self._processed_image = self._original_image[x_start_crop:x_end_crop,y_start_crop:y_end_crop]
        pass

    def _set_grayscale(self, image: TensorImage):
        pass
        """
        Metoda do ustawienia grayscale na obrazie zapisanym w klasie

        :param image:
        :type image:
        :return:
        :rtype:
        """



    def adjust_image(self, image: Tensor, crop_width: int, crop_length: int) -> Tensor:
        """


        :param image:
        :param crop_width:
        :param crop_length:
        :return:
        """
        self._load_image(image)
        self._crop_image(crop_width,crop_length)
        
        return self._processed_image

