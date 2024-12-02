import time
import pathlib
import numpy as np

class Utils:

    tick_memory = np.zeros(15)  # Zmienna stosowana do stworzenia średniej kroczącej

    @classmethod
    def calculate_fps(cls) -> np.floating:
        """
        Metoda do obliczania wartości klatek na sekundę.

        :return: Ilość klatek na sekundę.
        :rtype: np.floating
        """

        current_tick = time.time()
        # Implementacja średniej kroczącej
        cls.tick_memory = np.roll(cls.tick_memory,-1)
        partial_fps = np.zeros(len(cls.tick_memory)-1)
        cls.tick_memory[-1] = current_tick
        # Obliczenie pośrednich wartości klatek na sekundę, a następnie obliczenie jej średniej
        for i in range(len(cls.tick_memory)-1):
            if cls.tick_memory[-1-i] != cls.tick_memory[-2-i]:
                partial_fps[i] = 1/(cls.tick_memory[-1-i] - cls.tick_memory[-2-i])

        fps = np.mean(partial_fps)
        return fps

    @classmethod
    def fix_pathlib(cls) -> type(pathlib.WindowsPath):
        """
        Obejście błędu, który uniemożliwia tworzenie instancji PosixPath na systemie operacyjnym
        Więcej na temat tego problemu na https://github.com/ultralytics/yolov5/issues/10240
        :return: Ścieżka PosixPath
        :rtype: WindowsPath
        """
        fixed_path = pathlib.WindowsPath
        return fixed_path

    @classmethod
    def plot_face(cls, parameter_calculator, face_plotter, face_mesh_coords):
        """
        Metoda do stworzenia wykresu 3d wskaźników na twarzy.

        :param parameter_calculator: Klasa do wyliczania współrzędnych na podstawie wyniku z mediapipe
        :type parameter_calculator: CoordinatesParser
        :param face_plotter: Klasa do tworzenia wykresów na podstawie wyniku z mediapipe
        :type face_plotter: FacePlotter
        :param face_mesh_coords: Wynik wykrycia wskaźników na twarzy z mediapipe
        :type face_mesh_coords: Union
        """

        # Zdobycie współrzędnych odpowiednich fragmentów twarzy
        coords_left_eye = parameter_calculator.find_left_eye(face_mesh_coords)
        coords_right_eye = parameter_calculator.find_right_eye(face_mesh_coords)
        coords_mouth = parameter_calculator.find_mouth(face_mesh_coords)
        coords_left_iris = parameter_calculator.find_left_iris(face_mesh_coords)
        coords_right_iris = parameter_calculator.find_right_iris(face_mesh_coords)

        # Uzyskanie listy współrzędnych w formie, która biblioteka matplotlib może przetworzyć
        x_list_1, y_list_1, z_list_1 = parameter_calculator.get_coordinates(coords_left_eye)
        x_list_2, y_list_2, z_list_2 = parameter_calculator.get_coordinates(coords_right_eye)
        x_list_3, y_list_3, z_list_3 = parameter_calculator.get_coordinates(coords_mouth)
        x_list_4, y_list_4, z_list_4 = parameter_calculator.get_coordinates(coords_left_iris)
        x_list_5, y_list_5, z_list_5 = parameter_calculator.get_coordinates(coords_right_iris)

        # Aktualizacja wykresy o obecną pozycję twarzy
        face_plotter.update_xyz_coords(x_list_1, y_list_1, z_list_1, "LEFT_EYE")
        face_plotter.update_xyz_coords(x_list_2, y_list_2, z_list_2, "RIGHT_EYE")
        face_plotter.update_xyz_coords(x_list_3, y_list_3, z_list_3, "MOUTH")
        face_plotter.update_xyz_coords(x_list_4, y_list_4, z_list_4, "LEFT_IRIS")
        face_plotter.update_xyz_coords(x_list_5, y_list_5, z_list_5, "RIGHT_IRIS")


    @classmethod
    def frozenset_to_list(cls, face_frag: frozenset) -> list:
        """
        Funkcja przekształcająca obiekt typu frozenset z biblioteki MediaPipe, zawierający indeksy określonego
        fragmentu twarzy, na listę sąsiadujących ze sobą punktów, które razem tworzą linię.

        :param face_frag: Zmienna pochodząca z biblioteki mediapipe z pliku face_mesh_connections
        :type face_frag: frozenset
        :return: Lista z liniami, które mogą być wykreślone przez Matplotlib
        :rtype: list
        """

        all_face_lines = list()

        # Dla każdego połączenia szuka, czy istnieje punkt, z którym może go połączyć, a jak nie to tworzy nową listę.
        for tuple_connection in face_frag:
            is_added = False
            exists_in_two = False
            add_loc = []

            # Sprawdzenie, czy można połączyć obecną linię, jeśli ona sąsiaduję z inną
            for index,face_line in enumerate(all_face_lines, start = 0):

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