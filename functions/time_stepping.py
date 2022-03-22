from phi import math, field

def determine_step(grad, gridSize, factor = 0.9):
    velocity_max = math.max(math.abs(grad))
    return factor * gridSize / velocity_max

def grad_upwind(phi_field):
    phi_grad = field.spatial_gradient(phi_field)

    backward = field.spatial_gradient(phi_field, difference = 'backward')
    forward = field.spatial_gradient(phi_field, difference = 'forward')
    grad_plus = math.sum((math.where(backward.values > 0., backward.values, 0.))**2, dim='vector')+\
                math.sum((math.where(forward.values < 0., forward.values, 0.))**2, dim='vector')
    grad_minus = math.sum((math.where(backward.values < 0., backward.values, 0.))**2, dim='vector')+\
                 math.sum((math.where(forward.values > 0., forward.values, 0.))**2, dim='vector')
        
    return math.sqrt(grad_plus), math.sqrt(grad_minus), phi_grad