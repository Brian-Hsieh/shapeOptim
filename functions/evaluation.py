import warnings
import numpy as np
from phi import math, field
from phi.field import CenteredGrid
from .smooth_fcn import heaviside_atan, dirac
from .interpolate import get_interpolated_pressure

def check_area(input_field: CenteredGrid, ref_area, rel_tolerance = 1e-3, grid_factor = 1):
    gridSize = input_field.dx.native()[0]
    binary_mask = heaviside_atan(-1*input_field, gridSize/grid_factor)
    area = math.sum(binary_mask.values)*(gridSize**2)
    rel_error = math.abs(area-ref_area)/ref_area
    return rel_error < rel_tolerance

def evaluate_force(phi_input, pressure, velocity, diffusivity, epsilon_factor = 1):

    pressure = pressure.with_values(math.stop_gradient(pressure.values))
    velocity = velocity.with_values(math.stop_gradient(velocity.values))
    
    phi_dirac = dirac(phi_input, 1/(epsilon_factor*pressure.dx.native()[0]))
    phi_grad = field.spatial_gradient(phi_input)
    
    pressure_drag_vec = get_interpolated_pressure(pressure) * 1 * phi_grad * phi_dirac * pressure.dx.native()[0]**2
    pressure_drag = math.sum(pressure_drag_vec.values, dim = 'x,y')

    du_dy = (field.spatial_gradient(velocity.vector[0])).vector[1] @ pressure
    dv_dx = (field.spatial_gradient(velocity.vector[1])).vector[0] @ pressure
    vorticity = dv_dx - du_dy
    viscous_drag_vector = vorticity * 1 * phi_grad * phi_dirac * 2 * pressure.dx.native()[0]**2
    viscous_drag = diffusivity * math.sum(viscous_drag_vector.values, dim='x,y') * [-1,1]

    return pressure_drag, viscous_drag

def evaluate_position(phi_input, gridSize):
    coord = phi_input.points
    mask = heaviside_atan(-1*phi_input, gridSize/100)
    center_geo = math.sum((mask * coord).values, dim='x,y') / math.sum(mask.values)
    return center_geo

####################### CONVERGENCE EVALUATION #######################

def check_convergence(record, cur_iter, convergence_window: int, tolerance):
    if len(record) < 2*convergence_window:
        raise IndexError('Convergence window is too big.')
    avg_old = np.mean(record[cur_iter-2*convergence_window:cur_iter-convergence_window])
    avg = np.mean(record[cur_iter-convergence_window:cur_iter])
    if np.abs((avg-avg_old)/avg_old) < tolerance:
        return True
    else:
        return False

def check_multiplier_convergence(total_loss, loss, window, tolerance = 1e-4):
    total_loss = np.array(total_loss)
    loss = np.array(loss)
    ratio = abs(np.mean((total_loss[-1*window:] - loss[-1*window:])/loss[-1*window:]))
    if ratio < tolerance:
        return True, ratio
    return False, ratio

class Scanner():
    def __init__(self, inspect_name, inspect_axis):
        self.__assign_col(inspect_name, inspect_axis)
        self.__num_sign_change = 0

    @property
    def num_sign_change(self):
        return self.__num_sign_change

    def update_num_sign_change(self, record, cur_iter, window_size = 3):
        if cur_iter > 1 and self.check_sign_change(record, cur_iter):
            if self.check_signs(record, cur_iter, window_size):
                print('enter check sign')
                self.__num_sign_change += 1
            else:
                print('enter restart')
                self.__num_sign_change = -1

    def restart(self):
        self.__num_sign_change = 0

    def check_sign_change(self, record, cur_iter):
        record_1, record_2 = record[cur_iter-1:cur_iter+1, self.__inspect_col]
        return record_1 * record_2 <= 0

    def check_signs(self, record, cur_iter, window_size = 3):
        assert window_size >= 2, 'window_size should be at least 2'
        if cur_iter+1 < window_size:
            return False
        *record_previous, record_end = record[cur_iter-window_size+1:cur_iter+1, self.__inspect_col]
        pre_signs = [i < 0 for i in record_previous]
        cur_sign = record_end < 0
        cur_check = [cur_sign ^ i for i in pre_signs]
        return all(cur_check)

    def check_same_direction(self, record, cur_iter):
        warnings.warn("This method is deprecated. Use check_signs instead", DeprecationWarning)
        record_0, record_1, record_2 = record[cur_iter-2:cur_iter+1, self.__inspect_col]
        return (record_1 - record_0) * (record_2 - record_1) > 0
    
    def __assign_col(self, name, axis):
        if name == 'pressure':
            if axis == 'x': self.__inspect_col = 0
            else: self.__inspect_col = 1
        else:
            if axis == 'x': self.__inspect_col = 1
            else: self.__inspect_col = 0