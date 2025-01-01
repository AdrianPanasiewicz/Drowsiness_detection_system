class FacePlotter:
    def __init__(self, figure, axes3d, canvas, root):
        """
        Inicjalizacja obiektu FacePlotter, odpowiedzialnego za wyświetlanie i
        aktualizację punktów na twarzy w przestrzeni 3D w interfejsie Tkinter.

        :param figure: Obiekt klasy Figure (Matplotlib), w którym rysowany jest wykres.
        :type figure: matplotlib.figure.Figure
        :param axes3d: Oś 3D (Axes3D) z biblioteki Matplotlib, na której będą rysowane punkty.
        :type axes3d: mpl_toolkits.mplot3d.axes3d.Axes3D
        :param canvas: Obiekt FigureCanvasTkAgg, umożliwiający osadzenie wykresu w Tkinter.
        :type canvas: matplotlib.backends.backend_tkagg.FigureCanvasTkAgg
        :param root: Główne okno (widget) Tkinter, wykorzystywane do wywoływania metody after().
        :type root: tkinter.Tk lub tkinter.Toplevel
        """

        # Słowniki przechowujące współrzędne X, Y oraz Z dla wykrywanych elementów twarzy
        self.x_dict_all = dict()
        self.y_dict_all = dict()
        self.z_dict_all = dict()

        # Referencje do obiektów Matplotlib i Tkinter
        self.fig = figure
        self.ax = axes3d
        self.canvas = canvas
        self.root = root

        # Flaga określająca, czy animacja (aktualizacja wykresu) jest uruchomiona
        self._animation_running = False

    def start_animation(self, interval=100):
        """
        Rozpoczyna animację wykresu w ustalonym odstępie czasowym (co 'interval' milisekund).
        Wywołuje wewnętrzną metodę _update_plot() przy pomocy metody Tkinter after().

        :param interval: Częstotliwość odświeżania w milisekundach (np. 100 = 10 FPS).
        :type interval: int
        """
        self._animation_running = True
        self._update_plot(interval)

    def stop_animation(self):
        """
        Zatrzymuje animację, uniemożliwiając dalsze aktualizacje wykresu.
        """
        self._animation_running = False

    def update_xyz_coords(self, x_list: list, y_list: list, z_list: list, name: str):
        """
        Aktualizuje lub dodaje nowe współrzędne (X, Y, Z) dla wybranego fragmentu twarzy.
        Jeżeli wpis pod kluczem 'name' nie istnieje, tworzy nowy; w przeciwnym razie aktualizuje istniejący.

        :param x_list: Lista współrzędnych X danego fragmentu twarzy.
        :type x_list: list
        :param y_list: Lista współrzędnych Y danego fragmentu twarzy.
        :type y_list: list
        :param z_list: Lista współrzędnych Z danego fragmentu twarzy.
        :type z_list: list
        :param name: Nazwa fragmentu twarzy (np. "LEFT_EYE", "MOUTH").
        :type name: str
        """
        self.x_dict_all.update({name: x_list})
        self.y_dict_all.update({name: y_list})
        self.z_dict_all.update({name: z_list})

    def _update_plot(self, interval=33):
        """
        Metoda wewnętrzna odpowiedzialna za cykliczną aktualizację wykresu. Czyści istniejący
        rysunek, ustawia parametry osi, a następnie rysuje nowe punkty na bazie danych ze
        słowników współrzędnych. Po wykonaniu rysowania, planuje kolejne wywołanie samej siebie
        po określonym czasie 'interval', o ile animacja nie została zatrzymana.

        :param interval: Częstotliwość aktualizacji w milisekundach.
        :type interval: int
        """
        # Sprawdzenie, czy animacja nie została zatrzymana
        if not self._animation_running:
            return

        # Czyszczenie obszaru rysowania i nadanie limitów osi
        self.ax.cla()
        self.ax.set(
            xlim=(0, 1),
            ylim=(-0.5, 0.5),
            zlim=(0, 1),
            xlabel='Width',
            ylabel='Depth',
            zlabel='Height'
        )

        # Rysowanie wszystkich punktów zarejestrowanych w słownikach
        for key in self.x_dict_all:
            # Dla każdej osoby w danym fragmencie twarzy
            for person_index in range(len(self.x_dict_all[key])):
                # Dla każdej 'linii' (łańcucha punktów) we współrzędnych
                for line_index in range(len(self.x_dict_all[key][person_index])):
                    line_color = self._select_color(key, person_index)
                    x_vals = self.x_dict_all[key][person_index][line_index]
                    y_vals = self.y_dict_all[key][person_index][line_index]
                    z_vals = self.z_dict_all[key][person_index][line_index]
                    self.ax.plot(x_vals, z_vals, y_vals, color=line_color)

        # Odświeżenie widoku w interfejsie
        self.canvas.draw()

        # Zaplanowanie kolejnej aktualizacji po 'interval' milisekundach
        self.root.after(interval, self._update_plot, interval)

    @staticmethod
    def _select_color(key: str, person_index: int) -> str:
        """
        Wybiera kolor (w formacie heksadecymalnym) na podstawie nazwy fragmentu twarzy
        (parametr 'key') oraz indeksu osoby, jeśli w kadrze może być wiele twarzy.

        :param key: Nazwa fragmentu twarzy (np. "LEFT_EYE", "MOUTH").
        :type key: str
        :param person_index: Indeks osoby, jeśli wykryto wiele osób (domyślnie 0 przy jednej twarzy).
        :type person_index: int
        :return: Kolor w formacie szesnastkowym np. "#ff0000".
        :rtype: str
        """
        # Przykładowe mapowanie kolorów w zależności od fragmentu twarzy i indeksu osoby
        if key.upper() == "LEFT_EYE":
            line_color = FacePlotter._format_rgb_string(
                15 + 50 * person_index,
                15 + 50 * person_index,
                160 - 50 * person_index
            )
        elif key.upper() == "RIGHT_EYE":
            line_color = FacePlotter._format_rgb_string(
                15 + 50 * person_index,
                15 + 50 * person_index,
                160 - 50 * person_index
            )
        elif key.upper() == "MOUTH":
            line_color = FacePlotter._format_rgb_string(
                160 - 50 * person_index,
                15 + 50 * person_index,
                15 + 50 * person_index
            )
        elif key.upper() == "LEFT_IRIS":
            line_color = FacePlotter._format_rgb_string(
                15,
                15 + 50 * person_index,
                15 + 50 * person_index
            )
        elif key.upper() == "RIGHT_IRIS":
            line_color = FacePlotter._format_rgb_string(
                15,
                15 + 50 * person_index,
                15 + 50 * person_index
            )
        else:
            line_color = FacePlotter._format_rgb_string(
                150 - 50 * person_index,
                15,
                150 - 50 * person_index
            )

        return line_color

    @staticmethod
    def _format_rgb_string(rval: int, gval: int, bval: int) -> str:
        """
        Konwertuje wartości składowych RGB (czerwonej, zielonej i niebieskiej) na format
        szesnastkowy (#RRGGBB), gdzie każda składowa mieści się w zakresie [0, 255].

        :param rval: Wartość składowej czerwonej.
        :type rval: int
        :param gval: Wartość składowej zielonej.
        :type gval: int
        :param bval: Wartość składowej niebieskiej.
        :type bval: int
        :return: Kolor w formacie szesnastkowym (np. "#0f00a3").
        :rtype: str
        """
        # Normalizacja wartości składowych (mod 256) i zamiana na zapis szesnastkowy bez prefiksu "0x"
        red = f"{rval % 256:02x}"
        green = f"{gval % 256:02x}"
        blue = f"{bval % 256:02x}"

        # Złożenie całości w formacie #RRGGBB
        line_color = f"#{red}{green}{blue}"
        return line_color
