
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from varname import nameof
from .dir_manager import cwd

class Plotter():

    def __init__(self, x, y, radius, path):
        self._x = x
        self._y = y
        self._radius = radius
        self._path = path
        self._fig = None

    def plot(self):
        self._fig.canvas.draw()

    def savefig(self, name:int):
        with cwd(self._path):
            self._fig.savefig('{:05d}'.format(name))

class FieldPlotter(Plotter):

    def __init__(self, x, y, phi_field, radius, velocity, pressure, path):
        super().__init__(x, y, radius, path)
        self._fig, (self.ax_velocity, self.ax_pressure) = plt.subplots(nrows=2)
        self.images = {}
        self.__init_plot(velocity, nameof(velocity), phi_field, self.ax_velocity)
        self.__init_plot(pressure, nameof(pressure), phi_field, self.ax_pressure)
        self._fig.tight_layout()
        self.plot()

    def update(self, velocity, pressure, phi_field):
        self.__update_plot(self.images[nameof(velocity)], velocity, phi_field, self.ax_velocity)
        self.__update_plot(self.images[nameof(pressure)], pressure, phi_field, self.ax_pressure)

    def __init_plot(self, field, field_name, phi_field, ax):
        ax.set_title('Contour in {}'.format(field_name))
        self.images[field_name] = ax.imshow(field, origin = 'lower', extent=[-1,3,-1,1])
        ax.contour(self._x, self._y, phi_field, 0, colors='w')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cb = self._fig.colorbar(self.images[field_name], cax = cax)
        ax.set(xlim=[-2*self._radius,4*self._radius], ylim=[-2*self._radius,2*self._radius])

    def __update_plot(self, image, field, phi_field, ax):
        for line in ax.lines + ax.collections:
            line.remove()
        image.set_array(field)
        image.set_clim(vmin = np.min(field), vmax = np.max(field))
        ax.contour(self._x, self._y, phi_field, 0, colors ='w')
        self.plot()

class MaskPlotter(Plotter):

    def __init__(self, x, y, phi_field, radius, phi_mask, path):
        super().__init__(x, y, radius, path)
        self._fig, self.ax = plt.subplots()
        self.__init_plot(phi_mask, phi_field, self.ax)
        self.plot()

    def update(self, mask, phi_field):
        for line in self.ax.lines + self.ax.collections:
            line.remove()
        self.im.set_array(mask)
        self.ax.contour(self._x, self._y, phi_field, 0, colors='b', linewidths = 1.)
        self.plot()

    def __init_plot(self, mask, phi_field, ax):
        ax.set_title("Contour of zero level set")
        self.im = ax.imshow(mask, extent = [-1,3,-1,1], cmap = 'Wistia', origin='lower')
        self._fig.colorbar(self.im)
        ax.contour(self._x, self._y, phi_field, 0, colors='b', linewidths = 1.5)
        ax.set(xlim=[-1.5*self._radius,1.5*self._radius], ylim=[-1.5*self._radius,1.5*self._radius])
        ax.set_aspect('equal', adjustable='box')