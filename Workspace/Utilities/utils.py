import pathlib
import time
import numpy as np

class Utils:
    """
    Zapewnia metody pomocnicze do różnych zadań obliczeniowych i wizualizacyjnych.
    Ta klasa jest zaprojektowana do przetwarzania danych w konkretnym kontekście,
    takim jak obliczanie FPS, pakowanie danych, renderowanie współrzędnych twarzy
    oraz konwersja frozenset na listy.

    :ivar tick_memory: Utrzymuje ruchomą pamięć znaczników czasu, służącą do obliczania FPS.
    :type tick_memory: numpy.ndarray
    """

    tick_memory = np.zeros(15)  # Zmienna stosowana do stworzenia średniej kroczącej

    @classmethod
    def calculate_fps(cls) -> np.floating:
        """
        Calculates and returns the frames per second (FPS) based on the tick timestamps
        stored in the memory. This is implemented using a moving average method.
        The method calculates intermediate FPS values for each time interval in the tick
        memory, then averages these values to produce the final FPS.

        :rtype: numpy.floating
        :return: Calculated frames per second as a floating-point number.
        """

        current_tick = time.time()
        # Implementacja średniej kroczącej
        cls.tick_memory = np.roll(cls.tick_memory, -1)
        partial_fps = np.zeros(len(cls.tick_memory) - 1)
        cls.tick_memory[-1] = current_tick
        # Obliczenie pośrednich wartości klatek na sekundę, a następnie obliczenie jej średniej
        for i in range(len(cls.tick_memory) - 1):
            if cls.tick_memory[-1 - i] != cls.tick_memory[-2 - i]:
                partial_fps[i] = 1 / (cls.tick_memory[-1 - i] - cls.tick_memory[-2 - i])

        fps = np.mean(partial_fps)
        return fps

    @classmethod
    def fix_pathlib(cls) -> type(pathlib.WindowsPath):
        """
        Oblicza i zwraca liczbę klatek na sekundę (FPS) na podstawie znaczników czasu
        zapisanych w pamięci. Implementacja opiera się na metodzie średniej ruchomej.

        Metoda oblicza pośrednie wartości FPS dla każdego przedziału czasowego
        z pamięci znaczników, a następnie uśrednia te wartości, aby uzyskać ostateczne FPS.

        :rtype: numpy.floating
        :return: Obliczona liczba klatek na sekundę
        """
        fixed_path = pathlib.WindowsPath
        return fixed_path


    @classmethod
    def render_face_coordinates(cls, coordinates_parser, face_plt, face_mesh_coords):
        """
        Przetwarza i renderuje współrzędne twarzy poprzez wyodrębnienie konkretnych cech twarzy
        oraz aktualizację wykresu współrzędnymi, aby wizualizować strukturę twarzy w przestrzeni 3D.
        Metoda identyfikuje punkty charakterystyczne twarzy, takie jak oczy, usta i tęczówki,
        konwertuje ich współrzędne na formę zgodną z wykresem, a następnie aktualizuje system wykresów
        z bieżącą pozycją.

        :param coordinates_parser: Obiekt umożliwiający wyodrębnianie i konwersję współrzędnych cech twarzy.
        :type coordinates_parser: CoordinatesParser
        :param face_plt: Obiekt odpowiedzialny za aktualizację i renderowanie współrzędnych twarzy na wykresie.
        :type face_plt: FacePlotter
        :param face_mesh_coords: Pełny zestaw współrzędnych siatki twarzy, z którego wyodrębniane są konkretne cechy.
        :type face_mesh_coords: list
        :return: Zaktualizowany wykres z uwzględnieniem pozycji struktury twarzy.
        :rtype: None
        """

        # Zdobycie współrzędnych odpowiednich fragmentów twarzy
        coords_left_eye = coordinates_parser.find_left_eye(face_mesh_coords)
        coords_right_eye = coordinates_parser.find_right_eye(face_mesh_coords)
        coords_mouth = coordinates_parser.find_mouth(face_mesh_coords)
        coords_left_iris = coordinates_parser.find_left_iris(face_mesh_coords)
        coords_right_iris = coordinates_parser.find_right_iris(face_mesh_coords)

        # Uzyskanie listy współrzędnych w formie, która biblioteka matplotlib może przetworzyć
        x_list_1, y_list_1, z_list_1 = coordinates_parser.coords_to_plot_form(coords_left_eye)
        x_list_2, y_list_2, z_list_2 = coordinates_parser.coords_to_plot_form(coords_right_eye)
        x_list_3, y_list_3, z_list_3 = coordinates_parser.coords_to_plot_form(coords_mouth)
        x_list_4, y_list_4, z_list_4 = coordinates_parser.coords_to_plot_form(coords_left_iris)
        x_list_5, y_list_5, z_list_5 = coordinates_parser.coords_to_plot_form(coords_right_iris)

        # Aktualizacja wykresy o obecną pozycję twarzy
        face_plt.update_xyz_coords(x_list_1, y_list_1, z_list_1, "LEFT_EYE")
        face_plt.update_xyz_coords(x_list_2, y_list_2, z_list_2, "RIGHT_EYE")
        face_plt.update_xyz_coords(x_list_3, y_list_3, z_list_3, "MOUTH")
        face_plt.update_xyz_coords(x_list_4, y_list_4, z_list_4, "LEFT_IRIS")
        face_plt.update_xyz_coords(x_list_5, y_list_5, z_list_5, "RIGHT_IRIS")

    @classmethod
    def frozenset_to_list(cls, frozen_connections: frozenset) -> list:
        """
        Konwertuje frozenset połączeń na listę połączonych linii. Każde połączenie w frozenset
        powinno być tuple dwóch elementów, reprezentującą punkty końcowe. Funkcja przetwarza
        te połączenia, aby pogrupować połączone elementy w oddzielne listy, łącząc połączenia
        tam, gdzie to możliwe, aby utworzyć dłuższe, ciągłe linie.

        :param frozen_connections: Zbiór frozenset tuples, gdzie każda krotka reprezentuje
            połączenie między dwoma punktami.
        :type frozen_connections: frozenset
        :return: Posortowana lista zawierająca listy połączonych linii. Każda wewnętrzna lista
            reprezentuje sekwencję połączonych punktów.
        :rtype: list
        """

        all_face_lines = list()

        # Dla każdego połączenia szuka, czy istnieje punkt, z którym może go połączyć, a jak nie to tworzy nową listę.
        for tuple_connection in frozen_connections:
            is_added = False
            exists_in_two = False
            add_loc = []

            # Sprawdzenie, czy można połączyć obecną linię, jeśli ona sąsiaduję z inną
            for index, face_line in enumerate(all_face_lines, start=0):

                if tuple_connection[0] == face_line[-1]:
                    if is_added:
                        exists_in_two = True
                        add_loc.append(index)
                        break
                    face_line.append(tuple_connection[1])
                    is_added = True
                    add_loc.append(index)

                elif tuple_connection[1] == face_line[0]:
                    if is_added:
                        exists_in_two = True
                        add_loc.append(index)
                        break
                    face_line.insert(0, tuple_connection[0])
                    is_added = True
                    add_loc.append(index)

            # Funkcja tworzy nową listę w przypadku, gdy nic nie znalazł
            if not is_added:
                all_face_lines.append(list(tuple_connection))

            # Jeśli linię można połączyć z dwoma już istniejącymi listami, to te listy łączy
            if exists_in_two:
                line2 = all_face_lines.pop(add_loc[1])
                line1 = all_face_lines.pop(add_loc[0])

                # print(f"Połączenie \t\t{tuple_connection}")
                # print(f"Indeksy: \t\t{add_loc}")
                # print(f"Lista 1: \t\t{line1}")
                # print(f"Lista 2: \t\t{line2}")

                if line1[-1] == line2[0]:
                    joint_line = line1[:-1] + line2
                    all_face_lines.append(joint_line)
                    # print(f"Wynik: \t\t{joint_line}\n")

                if line1[0] == line2[-1]:
                    joint_line = line2[:-1] + line1
                    all_face_lines.append(joint_line)
                    # print(f"Result: \t\t{joint_line}\n")

        return sorted(all_face_lines)
