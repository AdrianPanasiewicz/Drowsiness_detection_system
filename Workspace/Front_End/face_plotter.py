import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

# Need to fix this class or implement this class again.

class FacePlotter:
    def __init__(self):
        """

        """
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(projection='3d')
        self.x_dict_all = dict()
        self.y_dict_all = dict()
        self.z_dict_all = dict()

        # self.ax.set(xlim=(0.45, 0.65), ylim=(-0.08, 0.08), zlim=(0.45, 0.65),
        #         xlabel='Width', ylabel='Depth', zlabel='Height')
        #
        # self.ani = FuncAnimation(self.fig, self.animate, interval=33)
        # plt.show(block = False)

        Plot_thread = threading.Thread(target=self.plot_temp, daemon=True)
        Plot_thread.start()


    def animate(self, frame):
        """

        :param frame:
        :type frame:
        :return:
        :rtype:
        """
        self.ax.cla()
        self.ax.set(xlim=(0.45, 0.65), ylim=(-0.08, 0.08), zlim=(0.45, 0.65),
                xlabel='Width', ylabel='Depth', zlabel='Height')
        for key in self.x_dict_all:
            self.ax.plot(self.x_dict_all[key], self.z_dict_all[key], self.y_dict_all[key])

    def update_xyz_coords(self, x_list: list, y_list: list, z_list: list, name: str):
        """

        :param x_list:
        :type x_list:
        :param y_list:
        :type y_list:
        :param z_list:
        :type z_list:
        :param name:
        :type name:
        :return:
        :rtype:
        """
        self.x_dict_all.update({name: x_list})
        self.y_dict_all.update({name: y_list})
        self.z_dict_all.update({name: z_list})


    def plot_temp(self):
        """

        :return:
        :rtype:
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.show(block = False)

        while True:
            ax.cla()
            ax.set(xlim=(0, 1), ylim=(-0.5, 0.5), zlim=(0, 1),
                    xlabel='Width', ylabel='Depth', zlabel='Height')

            for key in self.x_dict_all:
                if len(key) > 0:
                    for person_index in range(len(self.x_dict_all[key])):
                        for line_index in range(len(self.x_dict_all[key][person_index])):
                            line_color = self.select_color(key, person_index)
                            x_vals = self.x_dict_all[key][person_index][line_index]
                            y_vals = self.y_dict_all[key][person_index][line_index]
                            z_vals = self.z_dict_all[key][person_index][line_index]
                            ax.plot(x_vals, z_vals, y_vals, color=line_color)
                else:
                    ax.plot([],[],[])

            # plt.show()
            plt.pause(0.1)

    @staticmethod
    def select_color(key: str, person_index: int) -> str:
        """

        :param key:
        :type key:
        :param person_index:
        :type person_index:
        :return:
        :rtype:
        """
        if key.upper() == "LEFT_EYE":
            line_color = FacePlotter.format_rgb_string(15 + 50 * person_index,15 + 50 * person_index , 160 - 50 * person_index)
        elif key.upper() == "RIGHT_EYE":
            line_color = FacePlotter.format_rgb_string(15 + 50 * person_index,15 + 50 * person_index , 160 - 50 * person_index)
        elif key.upper() == "MOUTH":
            line_color = FacePlotter.format_rgb_string(160 - 50 * person_index,15 + 50 * person_index , 15 + 50 * person_index)
        elif key.upper() == "LEFT_IRIS":
            line_color = FacePlotter.format_rgb_string(15,15 + 50 * person_index , 15 + 50 * person_index)
        elif key.upper() == "RIGHT_IRIS":
            line_color = FacePlotter.format_rgb_string(15 ,15 + 50 * person_index , 15 + 50 * person_index)
        else:
            line_color = FacePlotter.format_rgb_string(150 - 50 * person_index,15 , 150 - 50 * person_index)

        return line_color

    @staticmethod
    def format_rgb_string(rval: int, gval: int, bval: int) -> str:
        """

        :param rval:
        :type rval:
        :param gval:
        :type gval:
        :param bval:
        :type bval:
        :return:
        :rtype:
        """
        red = str(hex(rval % 256))[2:]
        green = str(hex(gval % 256))[2:]
        blue = str(hex(bval % 256))[2:]

        if len(red) == 1:
            red = '0' + red
        if len(green) == 1:
            green = '0' + green
        if len(blue) == 1:
            blue = '0' + blue

        line_color = f"#{red}{green}{blue}"
        return line_color


