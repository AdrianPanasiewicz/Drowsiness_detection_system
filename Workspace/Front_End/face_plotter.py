import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class FacePlotter:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        # self.ani = FuncAnimation(self.fig, self.animate, interval=33)

        self.x_dict_all = dict()
        self.y_dict_all = dict()
        self.z_dict_all = dict()

        self.ax.set(xlim=(0.45, 0.65), ylim=(-0.08, 0.08), zlim=(0.45, 0.65),
                xlabel='Width', ylabel='Depth', zlabel='Height')

        plt.show()

    def animate(self):

        self.ax.cla()
        for key in self.x_dict_all:
            self.ax.plot(self.x_dict_all[key], self.y_dict_all[key], self.z_dict_all[key])

    def update_xyz_coords(self, x_list, y_list, z_list, name):

        self.x_dict_all.update({name: x_list})
        self.y_dict_all.update({name: y_list})
        self.z_dict_all.update({name: z_list})
