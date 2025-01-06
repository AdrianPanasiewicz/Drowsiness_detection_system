import pathlib
import time
import numpy as np
from typing import List, Type, Union


class Utils:
    """
    Klasa Utils dostarcza metody pomocnicze przydatne w różnych etapach
    przetwarzania danych związanych z analizą twarzy i obliczaniem
    wskaźników takich jak FPS, a także metody do konwersji struktur danych.

    :ivar tick_memory: Tablica (ruchoma pamięć) wykorzystywana do obliczania
                       średniej liczby klatek na sekundę (FPS).
    :type tick_memory: numpy.ndarray
    """

    # Tablica do obliczania średniej ruchomej FPS (15 ostatnich znaczników czasu).
    tick_memory: np.ndarray = np.zeros(15)

    @classmethod
    def calculate_fps(cls) -> float:
        """
        Oblicza i zwraca liczbę klatek na sekundę (FPS) na podstawie znaczników czasu
        przechowywanych w tablicy cls.tick_memory. Wykorzystuje w tym celu metodę
        średniej kroczącej. Dla każdej pary kolejnych znaczników obliczana jest
        tymczasowa wartość FPS, a następnie wyniki te są uśredniane, by uzyskać
        końcową wartość.

        :rtype: float
        :return: Obliczona liczba klatek na sekundę (FPS).
        """
        current_tick = time.time()

        # Przesunięcie zawartości tablicy w lewo o 1 (usuwa najstarszy znacznik, dopisuje najnowszy).
        cls.tick_memory = np.roll(cls.tick_memory, -1)
        cls.tick_memory[-1] = current_tick

        partial_fps = np.zeros(len(cls.tick_memory) - 1)
        # Dla każdej pary (tick_memory[i], tick_memory[i+1]) wyliczamy FPS,
        # a następnie tworzymy średnią z otrzymanych wartości.
        for i in range(len(cls.tick_memory) - 1):
            if cls.tick_memory[-1 - i] != cls.tick_memory[-2 - i]:
                diff = cls.tick_memory[-1 - i] - cls.tick_memory[-2 - i]
                if diff != 0:
                    partial_fps[i] = 1.0 / diff

        fps = float(np.mean(partial_fps))
        return fps

    @classmethod
    def fix_pathlib(cls) -> Type[pathlib.WindowsPath]:
        """
        Przykładowa metoda zwracająca typ pathlib.WindowsPath. W razie potrzeby
        można tu zawrzeć logikę modyfikującą lub rozszerzającą funkcjonalność
        ścieżek systemu plików w systemie Windows.

        :rtype: Type[pathlib.WindowsPath]
        :return: Typ pathlib.WindowsPath, który można wykorzystać do dalszej pracy z plikami.
        """
        fixed_path = pathlib.WindowsPath
        return fixed_path

    @classmethod
    def render_face_coordinates(
        cls,
        coordinates_parser: "CoordinatesParser",
        face_plt: "FacePlotter",
        face_mesh_coords: list
    ) -> None:
        """
        Wyodrębnia konkretne cechy twarzy (oczy, usta, tęczówki) z przekazanej
        listy współrzędnych siatki twarzy, a następnie aktualizuje obiekt
        odpowiadający za rysowanie (face_plt), aby zwizualizować twarz w 3D.

        :param coordinates_parser: Obiekt realizujący funkcje wyszukiwania
                                   i konwersji współrzędnych cech twarzy.
        :type coordinates_parser: CoordinatesParser
        :param face_plt: Obiekt odpowiedzialny za aktualizację i rysowanie
                         współrzędnych twarzy na wykresie 3D.
        :type face_plt: FacePlotter
        :param face_mesh_coords: Pełna lista współrzędnych siatki twarzy (zazwyczaj 468 punktów),
                                 z której wyodrębniamy poszczególne fragmenty (np. oczy, usta, tęczówki).
        :type face_mesh_coords: list
        :return: Nic nie zwraca. Efekt widoczny na wykresie.
        :rtype: None
        """

        # Pobranie współrzędnych (list punktów) dla konkretnych fragmentów twarzy.
        coords_left_eye = coordinates_parser.find_left_eye(face_mesh_coords)
        coords_right_eye = coordinates_parser.find_right_eye(face_mesh_coords)
        coords_mouth = coordinates_parser.find_mouth(face_mesh_coords)
        coords_left_iris = coordinates_parser.find_left_iris(face_mesh_coords)
        coords_right_iris = coordinates_parser.find_right_iris(face_mesh_coords)

        # Konwersja surowych współrzędnych na listy, które mogą być rysowane w Matplotlib.
        x_list_1, y_list_1, z_list_1 = coordinates_parser.coords_to_plot_form(coords_left_eye)
        x_list_2, y_list_2, z_list_2 = coordinates_parser.coords_to_plot_form(coords_right_eye)
        x_list_3, y_list_3, z_list_3 = coordinates_parser.coords_to_plot_form(coords_mouth)
        x_list_4, y_list_4, z_list_4 = coordinates_parser.coords_to_plot_form(coords_left_iris)
        x_list_5, y_list_5, z_list_5 = coordinates_parser.coords_to_plot_form(coords_right_iris)

        # Aktualizacja wykresu o najnowsze współrzędne (np. do rysowania linii oczu, ust, itd.).
        face_plt.update_xyz_coords(x_list_1, y_list_1, z_list_1, "LEFT_EYE")
        face_plt.update_xyz_coords(x_list_2, y_list_2, z_list_2, "RIGHT_EYE")
        face_plt.update_xyz_coords(x_list_3, y_list_3, z_list_3, "MOUTH")
        face_plt.update_xyz_coords(x_list_4, y_list_4, z_list_4, "LEFT_IRIS")
        face_plt.update_xyz_coords(x_list_5, y_list_5, z_list_5, "RIGHT_IRIS")

    @classmethod
    def frozenset_to_list(cls, frozen_connections: frozenset[tuple[int, int]]) -> List[List[int]]:
        """
        Konwertuje zbiór (frozenset) dwuelementowych krotek (reprezentujących
        połączenia między punktami) na listę list, gdzie każda wewnętrzna lista
        przedstawia ciąg połączonych punktów. Pozwala to w efekcie na łatwiejsze
        rysowanie tych połączeń jako linii na wykresie.

        :param frozen_connections: Zbiór krotek (np. {(1,2), (2,3), (4,5)...}),
                                   gdzie każda krotka definiuje połączenie między dwoma punktami.
        :type frozen_connections: frozenset[tuple[int,int]]
        :return: Lista list punktów, w której każdy element odpowiada ciągłej linii
                 utworzonej przez możliwe do połączenia krotki.
        :rtype: List[List[int]]
        """

        # Lista zawierająca ostateczne linie, gdzie każda linia to lista połączonych ze sobą indeksów punktów.
        all_face_lines = []

        # Dla każdego połączenia sprawdzamy, czy można je dołączyć do już istniejącej linii
        # lub czy musimy utworzyć nową.
        for tuple_connection in frozen_connections:
            is_added = False
            exists_in_two = False
            add_loc = []

            # Próba włączenia krotki do istniejących linii w all_face_lines.
            for index, face_line in enumerate(all_face_lines):
                # Jeśli koniec obecnej linii pokrywa się z początkiem krotki,
                # dołączamy drugi element krotki do linii.
                if tuple_connection[0] == face_line[-1]:
                    if is_added:
                        exists_in_two = True
                        add_loc.append(index)
                        break
                    face_line.append(tuple_connection[1])
                    is_added = True
                    add_loc.append(index)

                # Alternatywnie, jeśli początek linii pokrywa się z drugim elementem krotki,
                # dodajemy pierwszy element krotki na początek linii.
                elif tuple_connection[1] == face_line[0]:
                    if is_added:
                        exists_in_two = True
                        add_loc.append(index)
                        break
                    face_line.insert(0, tuple_connection[0])
                    is_added = True
                    add_loc.append(index)

            # Jeśli nie udało się dołączyć krotki do żadnej z istniejących linii,
            # tworzymy nową linię z krotki.
            if not is_added:
                all_face_lines.append(list(tuple_connection))

            # W sytuacji, gdy krotka pasuje do dwóch różnych linii, łączymy te linie w jedną.
            if exists_in_two:
                line2 = all_face_lines.pop(add_loc[1])
                line1 = all_face_lines.pop(add_loc[0])

                # Jeśli koniec line1 zbiega się z początkiem line2, sklejamy je w jedną.
                if line1[-1] == line2[0]:
                    joint_line = line1[:-1] + line2
                    all_face_lines.append(joint_line)

                # Jeśli natomiast początek line1 zbiega się z końcem line2,
                # sklejamy line2 z line1.
                if line1[0] == line2[-1]:
                    joint_line = line2[:-1] + line1
                    all_face_lines.append(joint_line)

        # Sortujemy, by uzyskać powtarzalny (deterministyczny) układ linii (np. do debugowania).
        return sorted(all_face_lines)
