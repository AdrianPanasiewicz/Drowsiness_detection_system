import cv2
import numpy as np
from mediapipe import solutions
from typing import Tuple, Any


class ImageProcessor:
    """
    Klasa odpowiedzialna za wstępne przetwarzanie obrazu, w tym m.in.
    wycinanie (crop), konwersję do skali szarości oraz wykrywanie siatki twarzy (face mesh).
    """

    def __init__(self) -> None:
        """
        Inicjalizuje obiekt ImageProcessor, definiując narzędzia z biblioteki
        MediaPipe (drawing_utils, face_mesh) oraz parametry rysowania landmarków.
        """
        self._mp_draw = solutions.drawing_utils
        self._mp_face_mesh = solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self._draw_spec = self._mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    @staticmethod
    def _crop_image(image: np.ndarray, crop_width: int, crop_height: int) -> np.ndarray:
        """
        Przycina obraz do zadanej szerokości i wysokości, wycinając środek kadru.

        :param image: Obraz w formacie zgodnym z OpenCV (np. BGR).
        :type image: np.ndarray
        :param crop_width: Docelowa szerokość w pikselach.
        :type crop_width: int
        :param crop_height: Docelowa wysokość w pikselach.
        :type crop_height: int
        :return: Przycięty obraz o wymiarach (crop_width, crop_height).
        :rtype: np.ndarray
        """
        x_size = image.shape[0]
        y_size = image.shape[1]

        x_start_crop = (x_size - crop_width) // 2
        x_end_crop = (x_size + crop_width) // 2
        y_start_crop = (y_size - crop_height) // 2
        y_end_crop = (y_size + crop_height) // 2

        cropped_image = image[x_start_crop:x_end_crop, y_start_crop:y_end_crop]
        return cropped_image

    @staticmethod
    def _set_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Konwertuje obraz z przestrzeni barw BGR do skali szarości (grayscale).

        :param image: Obraz w formacie BGR (np. załadowany przez OpenCV).
        :type image: np.ndarray
        :return: Obraz w skali szarości.
        :rtype: np.ndarray
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _find_face_mesh(self, image: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Wykrywa siatkę twarzy (face mesh) na obrazie i nanosi wykryte landmarki.
        Do przetwarzania obrazu używany jest tryb RGB (konwersja z BGR).

        :param image: Obraz w formacie BGR.
        :type image: np.ndarray
        :return: Krotka (image, results), gdzie:
                 - image (np.ndarray): Obraz w BGR z zaznaczonymi landmarkami twarzy (jeśli wykryto),
                 - results: Wynik przetwarzania z MediaPipe, zawierający m.in. multi_face_landmarks.
        :rtype: tuple
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self._mp_draw.draw_landmarks(
                    image,
                    face_landmarks,
                    self._mp_face_mesh.FACEMESH_CONTOURS,
                    self._draw_spec,
                    self._draw_spec
                )

        return image, results

    def crop_and_convert_to_gray(
        self,
        image: np.ndarray,
        crop_width: int,
        crop_height: int
    ) -> np.ndarray:
        """
        Wykonuje wycięcie (crop) środka obrazu do zadanych wymiarów
        oraz konwersję wynikowego obrazu do skali szarości.

        :param image: Obraz w formacie np. BGR.
        :type image: np.ndarray
        :param crop_width: Docelowa szerokość przyciętego obrazu.
        :type crop_width: int
        :param crop_height: Docelowa wysokość przyciętego obrazu.
        :type crop_height: int
        :return: Obraz w skali szarości po operacji crop.
        :rtype: np.ndarray
        """
        cropped_image = self._crop_image(image, crop_width, crop_height)
        gray_image = self._set_grayscale(cropped_image)
        return gray_image

    def process_face_image(self, image: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Wykrywa siatkę twarzy (face mesh) na przekazanym obrazie i rysuje landmarki.
        Zwraca również wyniki obliczeń MediaPipe w postaci obiektu zawierającego
        m.in. listę wykrytych twarzy i ich punktów (multi_face_landmarks).

        :param image: Obraz w formacie BGR, na którym zostanie wykryta twarz.
        :type image: np.ndarray
        :return: Krotka (processed_image, face_mesh_coords), gdzie:
                 - processed_image (np.ndarray): Obraz z zaznaczonymi punktami twarzy,
                 - face_mesh_coords: Obiekt MediaPipe z informacjami o wykrytych landmarkach.
        :rtype: tuple
        """
        processed_image, face_mesh_coords = self._find_face_mesh(image)
        return processed_image, face_mesh_coords
