class FacePlotter:
    def __init__(self, figure, axes3d, canvas, root):
        """
        Inicjalizacja klasy FacePlotter.
        """

        self.x_dict_all = dict()
        self.y_dict_all = dict()
        self.z_dict_all = dict()

        self.fig = figure
        self.ax = axes3d
        self.canvas = canvas
        self.root = root

        self._animation_running = False

    def start_animation(self, interval=100):
        self._animation_running = True
        self._update_plot(interval)

    def stop_animation(self):
        self._animation_running = False

    def update_xyz_coords(self, x_list: list, y_list: list, z_list: list, name: str):
        """
        Metoda do zaktualizowania wartości współrzędnych danego fragmentu na twarzy w tej klasie, a jeśli nie istnieje
        to stworzenie nowego rekordu do ich zapisu.
        :param x_list: Lista współrzędnych X dla wskaźników na twarzy
        :type x_list: list
        :param y_list:Lista współrzędnych Y dla wskaźników na twarzy
        :type y_list: list
        :param z_list: Lista współrzędnych Z dla wskaźników na twarzy
        :type z_list: list
        :param name: Nazwa aktualizowanego fragmentu twarzy
        :type name: str
        """

        self.x_dict_all.update({name: x_list})
        self.y_dict_all.update({name: y_list})
        self.z_dict_all.update({name: z_list})

    def _update_plot(self, interval=33):
        """

        """
        if not self._animation_running:
            return

        self.ax.cla()
        self.ax.set(
            xlim=(0, 1),
            ylim=(-0.5, 0.5),
            zlim=(0, 1),
            xlabel='Width',
            ylabel='Depth',
            zlabel='Height'
        )

        for key in self.x_dict_all:
            for person_index in range(len(self.x_dict_all[key])):
                for line_index in range(len(self.x_dict_all[key][person_index])):
                    line_color = self._select_color(key, person_index)
                    x_vals = self.x_dict_all[key][person_index][line_index]
                    y_vals = self.y_dict_all[key][person_index][line_index]
                    z_vals = self.z_dict_all[key][person_index][line_index]
                    self.ax.plot(x_vals, z_vals, y_vals, color=line_color)

        self.canvas.draw()
        self.root.after(interval, self._update_plot, interval)

    @staticmethod
    def _select_color(key: str, person_index: int) -> str:
        """
        Metoda odpowiedzialna za wybór koloru, w którym dana część twarzy jest rysowana.
        :param key: Nazwa fragmentu twarzy
        :type key: str
        :param person_index: Indeks osoby, jeśli jest więcej niż jedna osoba obecna na kamerze.
        :type person_index: int
        :return: String z odpowiednim kolorem dla fragmentu twarzy
        :rtype: str
        """
        if key.upper() == "LEFT_EYE":
            line_color = FacePlotter._format_rgb_string(15 + 50 * person_index, 15 + 50 * person_index,
                                                        160 - 50 * person_index)
        elif key.upper() == "RIGHT_EYE":
            line_color = FacePlotter._format_rgb_string(15 + 50 * person_index, 15 + 50 * person_index,
                                                        160 - 50 * person_index)
        elif key.upper() == "MOUTH":
            line_color = FacePlotter._format_rgb_string(160 - 50 * person_index, 15 + 50 * person_index,
                                                        15 + 50 * person_index)
        elif key.upper() == "LEFT_IRIS":
            line_color = FacePlotter._format_rgb_string(15, 15 + 50 * person_index, 15 + 50 * person_index)
        elif key.upper() == "RIGHT_IRIS":
            line_color = FacePlotter._format_rgb_string(15, 15 + 50 * person_index, 15 + 50 * person_index)
        else:
            line_color = FacePlotter._format_rgb_string(150 - 50 * person_index, 15, 150 - 50 * person_index)

        return line_color

    @staticmethod
    def _format_rgb_string(rval: int, gval: int, bval: int) -> str:
        """
        Metoda do utworzenia string rgb na podstawie żądanych wartości koloru czerwonego, zielonego i niebieskiego.
        :param rval: Wartość koloru czerwonego
        :type rval: int
        :param gval: Wartość koloru zielonego
        :type gval: int
        :param bval: Wartość koloru niebieskiego
        :type bval: int
        :return: String w formacie rgb
        :rtype: str
        """
        # Przekształcenie wartości całkowitych kolorów na odpowiadające im wartości hex, zapisane jako string
        red = str(hex(rval % 256))[2:]
        green = str(hex(gval % 256))[2:]
        blue = str(hex(bval % 256))[2:]

        # W przypadku, gdy wartość koloru można zapisać jako liczba jednocyfrowa w systemie szesnastkowym, to dopełnić
        # dodatkowym zerem
        if len(red) == 1:
            red = '0' + red
        if len(green) == 1:
            green = '0' + green
        if len(blue) == 1:
            blue = '0' + blue

        # Stworzenie string rgb
        line_color = f"#{red}{green}{blue}"
        return line_color
