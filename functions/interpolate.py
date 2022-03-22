from scipy.interpolate import NearestNDInterpolator as NNInp
import numpy as np
from phi import math

def get_interpolated_pressure(input_pressure, flatten_direction = 'F'):
    uninterpolated = math.where(input_pressure.values, input_pressure.values, np.nan).numpy('x,y').flatten(flatten_direction)
    inter_mask = math.where(input_pressure.values, 1., np.nan)
    x,y = math.unstack(input_pressure.points, dim = 'vector')
    x, y = x.numpy('x,y'), y.numpy('x,y')
    points = (input_pressure.points * inter_mask).numpy('x,y,vector')
    x_p, y_p = points[...,0].flatten(flatten_direction), points[...,1].flatten(flatten_direction)
    interp = NNInp(list(zip(x_p, y_p)), uninterpolated)
    interpolated = interp(x,y)
    return input_pressure.with_values(math.tensor(interpolated, input_pressure.shape))