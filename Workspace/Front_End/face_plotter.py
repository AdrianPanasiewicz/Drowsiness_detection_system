import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

# Need to fix this class or implement this class again.

class FacePlotter:
    def __init__(self):
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

        self.ax.cla()
        self.ax.set(xlim=(0.45, 0.65), ylim=(-0.08, 0.08), zlim=(0.45, 0.65),
                xlabel='Width', ylabel='Depth', zlabel='Height')
        for key in self.x_dict_all:
            self.ax.plot(self.x_dict_all[key], self.z_dict_all[key], self.y_dict_all[key])

    def update_xyz_coords(self, x_list, y_list, z_list, name):

        self.x_dict_all.update({name: x_list})
        self.y_dict_all.update({name: y_list})
        self.z_dict_all.update({name: z_list})


    def plot_temp(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.show(block = False)

        while True:
            ax.cla()
            ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1),
                    xlabel='Width', ylabel='Depth', zlabel='Height')

            for key in self.x_dict_all:
                for person_index in range(len(self.x_dict_all[key])):
                    ax.plot(self.x_dict_all[key][person_index], self.z_dict_all[key][person_index], self.y_dict_all[key][person_index])

            # plt.show()
            plt.pause(0.1)


