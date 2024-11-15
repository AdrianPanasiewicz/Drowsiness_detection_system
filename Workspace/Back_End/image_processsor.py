from fastai.vision.all import *
import cv2
from mediapipe import solutions

class ImageProcessor:
    """
    Klasa odpowiedzialna za wstępne przetwarzanie obrazu.
    """
    def __init__(self):

        self._mpDraw = solutions.drawing_utils
        self._mpFaceMesh = solutions.face_mesh
        self._faceMesh = self._mpFaceMesh.FaceMesh()
        self._drawSpec = self._mpDraw.DrawingSpec(thickness=1, circle_radius=1, color = (0, 0, 255))


    @staticmethod
    def _crop_image(image, crop_width: int, crop_height: int):
        """
        Metoda do wykonania operacji crop na obrazie zapisanym w klasie

        :param image: Obraz, który będzie przetworzony
        :type image: Union z OpenCV
        :param crop_width: Szerokość, z którą ma być wykonana operacja crop
        :type crop_width: int
        :param crop_height: Wysokość, z którą ma być wykonana operacja crop
        :type crop_width: int
        :return: Przycięty obraz
        :rtype: Union
        """
        # Uzyskanie wymiarów obrazu
        x_size = image.shape[0]
        y_size = image.shape[1]

        # Obliczenie wspolrzednych do wykonania operacji crop o wymiarach crop_width i crop_height
        x_start_crop = (x_size - crop_width)//2
        x_end_crop = (x_size + crop_width)//2

        y_start_crop = (y_size - crop_height)//2
        y_end_crop = (y_size + crop_height)//2

        # Wykonanie operacji crop
        cropped_image = image[x_start_crop:x_end_crop,y_start_crop:y_end_crop]

        return cropped_image


    @staticmethod
    def _set_grayscale(image):
        """
        Metoda do ustawienia grayscale na obrazie zapisanym w klasie

        :param image: Obraz, który będzie przetworzony
        :type image: Union z OpenCV
        :return: Obraz w skali grayscale
        :rtype: Union
        """

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray_image

    def _find_face_mesh(self, image):
        """
        Metoda do wykrywania lokalizacji

        :param image: Obraz, który będzie przetworzony
        :type image: Union z OpenCV
        :return: Obraz, na którym są zaznaczone wskaźniki na twarzy
        :rtype: Union z OpenCV
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._faceMesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._mpDraw.draw_landmarks(image, face_landmarks, self._mpFaceMesh.FACEMESH_CONTOURS,
                                            self._drawSpec, self._drawSpec)

        return image,results



    def preprocess_image1(self, image, crop_width: int, crop_length: int):
        """
        Publiczna metoda do wstępnego przetwarzania obrazu

        :param image: Obraz, który ma być przetworzony
        :type image: Union z OpenCV
        :param crop_width: Szerokość, z którą ma być wykonana operacja crop
        :type crop_width: int
        :param crop_length: Długość, z którą ma być wykonana operacja crop
        :type crop_length: int
        :return: Przetworzony obraz
        """

        # Wstępne przetworzenie obrazu
        cropped_image = self._crop_image(image, crop_width,crop_length)
        gray_image = self._set_grayscale(cropped_image)

        # Zwrócenie przetworzonego obrazu
        return gray_image

    def preprocess_image2(self, image):
        """
        Interfejs do przetwarzania landmarks na twarzy

        :param image: Obraz, który będzie przetworzony
        :type image: Union z OpenCV
        :return:
        :rtype:
        """

        # Wstępne przetworzenie obrazu
        processed_image, face_mesh_coords = self._find_face_mesh(image)

        return processed_image, face_mesh_coords
