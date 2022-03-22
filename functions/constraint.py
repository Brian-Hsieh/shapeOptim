import numpy as np
import matplotlib.pyplot as plt
from .smooth_fcn import *
from .evaluation import evaluate_position
from .time_stepping import determine_step
from .regularization import make_sdf

def constrain_area_adam(field, ref_area, rel_tolerance = 1e-3 ,lr = 1e-3, max_iteration = 500, grid_factor = 1):
    
    step = 0
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m = 0
    v = 0
    area_list=[]
    field_old = field    
    shift = math.tensor(0.)
    converged = False
    
    for i in range(max_iteration):
        field = field_old

        with math.record_gradients(shift):
            
            field += shift
            area_vector = heaviside_atan(-1*field, field.dx.native()[0]/grid_factor)
            area = math.sum(area_vector.values)*(field.dx.native()[0])**2
            loss = math.l2_loss(area - ref_area)
            grad = math.gradients(loss)
            
        area_list.append(area.numpy())
        
        rel_error = (area - ref_area) / ref_area
        if math.abs(rel_error).numpy() < rel_tolerance:
            converged = True
            return shift, area.numpy(), area_list, converged

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        step += 1
        bias_corr1 = (1 - beta1**step)
        bias_corr2 = (1 - beta2**step)
        step_size = lr / bias_corr1
        t_grad = m / (math.sqrt(v/bias_corr2) + epsilon) * step_size
        shift -= t_grad
        
    print("Area rel. error: {}".format(rel_error))
    
    # shift = math.tensor(0.)
    print('Max. iteration reached, but not converged. Set shift to last one.')
    return shift, area.numpy(), area_list, converged

def constrain_position(input_field, init_pos, gridSize, substeps = 10):
    #Heun's method update with multiple substeps
    cur_pos = evaluate_position(input_field, gridSize)
    velocity = init_pos - cur_pos
    dt = determine_step(velocity, gridSize, 1/substeps)
    for step in range(int(1/dt)):
        field_old = input_field
        field_grad = field.spatial_gradient(input_field)
        k1 = math.sum((velocity * field_grad).values, dim = 'vector')
        input_field -= k1 * dt / 2
        field_grad = field.spatial_gradient(input_field)
        k2 = math.sum((velocity * field_grad).values, dim = 'vector')
        input_field  = field_old - k2 * dt
    output_field = make_sdf(input_field)
    return output_field, evaluate_position(output_field, gridSize)