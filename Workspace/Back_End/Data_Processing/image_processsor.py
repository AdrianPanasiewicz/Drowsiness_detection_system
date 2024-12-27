import cv2
import numpy as np
from mediapipe import solutions


class ImageProcessor:
    """
    Klasa odpowiedzialna za wstępne przetwarzanie obrazu.
    """

    def __init__(self):
        """
        Konstruktor klasy ImageProcessor
        """
        self._mp_draw = solutions.drawing_utils
        self._mp_face_mesh = solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self._draw_spec = self._mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    @staticmethod
    def _crop_image(image, crop_width: int, crop_height: int) -> np.ndarray:
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
        x_start_crop = (x_size - crop_width) // 2
        x_end_crop = (x_size + crop_width) // 2

        y_start_crop = (y_size - crop_height) // 2
        y_end_crop = (y_size + crop_height) // 2

        # Wykonanie operacji crop
        cropped_image = image[x_start_crop:x_end_crop, y_start_crop:y_end_crop]

        return cropped_image

    @staticmethod
    def _set_grayscale(image) -> np.ndarray:
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
        :return: Obraz, na którym są zaznaczone wskaźniki na twarzy oraz wynik działania mediapipe
        :rtype: np.ndarray, Union z OpenCV
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(image_rgb)

        # Jeśli znaleziono twarz, to na obrazie zostaną zaznaczone jej wskaźniki
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._mp_draw.draw_landmarks(image, face_landmarks, self._mp_face_mesh.FACEMESH_CONTOURS, self._draw_spec,
                                             self._draw_spec)

        return image, results

    def crop_and_convert_to_gray(self, image, crop_width: int, crop_length: int) -> np.ndarray:
        """
        Metoda do wstępnego przetwarzania obrazu

        :param image: Obraz, który ma być przetworzony
        :type image: Union z OpenCV
        :param crop_width: Szerokość, z którą ma być wykonana operacja crop
        :type crop_width: int
        :param crop_length: Długość, z którą ma być wykonana operacja crop
        :type crop_length: int
        :return: Przetworzony obraz
        :rtype: np.ndarray
        """

        # Wstępne przetworzenie obrazu
        cropped_image = self._crop_image(image, crop_width, crop_length)
        gray_image = self._set_grayscale(cropped_image)

        # Zwrócenie przetworzonego obrazu
        return gray_image

    def process_face_image(self, image):
        """
        Interfejs do przetwarzania wskaźników na twarzy

        :param image: Obraz, który będzie przetworzony
        :type image: Union z OpenCV
        :return: Obraz, na którym są zaznaczone wskaźniki na twarzy oraz wynik działania mediapipe
        :rtype: np.ndarray, Union z OpenCV
        """

        # Wstępne przetworzenie obrazu
        processed_image, face_mesh_coords = self._find_face_mesh(image)

        # Zwrócenie przetworzonego obrazu
        return processed_image, face_mesh_coords
