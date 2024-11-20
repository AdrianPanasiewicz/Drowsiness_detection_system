import time
import pathlib
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS

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
    def fix_pathlib(cls):
        # A work-around for the error that PosixPath cannot be instantiated on your system
        # More about this issue on https://github.com/ultralytics/yolov5/issues/10240
        fixed_path = pathlib.WindowsPath
        return fixed_path

    @classmethod
    def frozenset_to_list(cls, face_frag: frozenset) -> list:
        """
        Funkcja do przetwarzania frozenset z biblioteki mediapipe z indeksami do danego fragmentu twarzy do listy
        z liniami.

        :param face_frag: Zmienna pochodząca z biblioteki mediapipe z pliku face_mesh_connections
        :type face_frag: frozenset
        :return: Lista z liniami, które mogą być wykreślone przez Matplotlib
        :rtype: list
        """

        all_lips_lines = list()

        # Dla każdego połączenia szuka, czy istnieje punkt, z którym może go połączyć, a jak nie to tworzy nową listę.
        for tuple_connection in face_frag:
            is_added = False
            exists_in_two = False
            add_loc = []

            for index,lips_line in enumerate(all_lips_lines, start = 0):

                if tuple_connection[0] == lips_line[-1]:
                    if is_added:
                        exists_in_two = True
                        add_loc.append(index)
                        break

                    lips_line.append(tuple_connection[1])
                    is_added = True
                    add_loc.append(index)


                elif tuple_connection[1] == lips_line[0]:
                    if is_added:
                        exists_in_two = True
                        add_loc.append(index)
                        break

                    lips_line.insert(0, tuple_connection[0])
                    is_added = True
                    add_loc.append(index)

            # Funkcja tworzy nową listę w przypadku, gdy nic nie znalazł
            if not is_added:
                all_lips_lines.append(list(tuple_connection))

            # Jeśli można utworzyć połączanie w dwóch listach, to je łączy
            if exists_in_two:
                line2 = all_lips_lines.pop(add_loc[1])
                line1 = all_lips_lines.pop(add_loc[0])

                # print(f"Połączenie \t\t{tuple_connection}")
                # print(f"Indeksy: \t\t{add_loc}")
                # print(f"Lista 1: \t\t{line1}")
                # print(f"Lista 2: \t\t{line2}")

                if line1[-1] == line2[0]:
                    joint_line = line1[:-1] + line2
                    all_lips_lines.append(joint_line)
                    # print(f"Wynik: \t\t{joint_line}\n")

                if line1[0] == line2[-1]:
                    joint_line = line2[:-1] + line1
                    all_lips_lines.append(joint_line)
                    # print(f"Result: \t\t{joint_line}\n")

        return sorted(all_lips_lines)


if __name__ == "__main__":
    ret = Utils.frozenset_to_list(FACEMESH_LIPS)
    print(ret)


