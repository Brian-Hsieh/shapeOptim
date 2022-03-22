from phi import math, field
from phi.geom import Phi
from phi.physics import advect, fluid, diffuse
from phi.physics._boundaries import Obstacle
from .time_stepping import grad_upwind
from .smooth_fcn import dirac, heaviside_atan
from .interpolate import get_interpolated_pressure
from .evaluation import evaluate_force

class GradientStepper():
    def __init__(self, phi_field, radius, diffusivity):
        self.grad_plus, self.grad_minus, _ = grad_upwind(phi_field)
        self.field = phi_field
        self._radius = radius
        self._diffusivity = diffusivity
        self._gridSize = self.field.dx.native()[0]

        self._total_loss = 0.
        self._grad = None
        self._viscous_record = None
        self._pressure_record = None

    @property
    def total_loss(self):
        return self._total_loss
    
    @property
    def grad(self):
        return self._grad
    
    @property
    def viscous_record(self):
        return self._viscous_record
    
    @property
    def pressure_record(self):
        return self._pressure_record

    def set_phi(self, phi_input):
        self.grad_plus, self.grad_minus, _ = grad_upwind(phi_input)
        self.field = phi_input
        self.zero_loss()

    def zero_loss(self):
        self._total_loss = 0.

class AccumulatorEnergy(GradientStepper):
    '''
    The accumulator accumulates the energy dissipation for several iterations
    
    set_obs() links the phi field and the flow field through solvers
    run() keeps accumulating the total force
    backward() performs backpropagation
    
    Args:
        phi_field: input phi field
        inflow_velocity: self-defined velocity in StaggeredGrid or 2 dimensional tensor specifying velocity in x and y axis
        INFLOW: inflow mask
        obs: obstacle in flow
        target: number of half vortex oscillating period to be accumulated, e.g. 2 for one period
    '''
    def __init__(self, phi_field, radius, diffusivity, inflow_velocity, INFLOW, DT, obs, target = 0):
        super().__init__(phi_field, radius, diffusivity)
        self.__inflow_velocity = inflow_velocity
        self.__INFLOW = INFLOW
        self.__dt = DT
        self.__obs = obs
        self.__target = target

    @property
    def target(self):
        return self.__target
        
    def set_obs(self):
        with math.record_gradients(self.field.values):
            self.__obs = self.__obs.copied_with(geometry = Phi(self.field))
        
    def run(self, velocity, pressure):
        
        with math.record_gradients(self.field.values):
            
            velocity = advect.semi_lagrangian(velocity, velocity, self.__dt)
            velocity = velocity * (1- self.__INFLOW) + self.__INFLOW * self.__inflow_velocity
            velocity = diffuse.explicit(velocity, self._diffusivity, self.__dt, substeps = 4)
            velocity, pressure = fluid.make_incompressible(velocity, 
                                                           obstacles = (self.__obs,), 
                                                           solve=math.Solve('auto', math.tensor(1e-3), math.tensor(0.), max_iterations = math.tensor(1e4), x0 = pressure, gradient_solve=math.Solve('auto', math.tensor(1e-5), math.tensor(1e-5), max_iterations = math.tensor(1e3))))

            grad_x = field.spatial_gradient(velocity.vector[0]).vector[1]
            grad_y = field.spatial_gradient(velocity.vector[1]).vector[0]
            self._total_loss += (math.l2_loss(grad_x) + math.l2_loss(grad_y))

        pressure_force, viscous_force = evaluate_force(self.field, pressure/self.__dt, velocity, self._diffusivity)
        self._viscous_record = viscous_force
        self._pressure_record = pressure_force
        
        return pressure, velocity
    
    def backward(self, velocity, pressure):
        with math.record_gradients(self.field.values):
            self._grad = math.gradients(self._total_loss)

        if self.__target != 0:
            self._grad *= self.__dt
            self._total_loss *= self.__dt

        self.field = self.field.with_values(math.stop_gradient(self.field.values))
        pressure = pressure.with_values(math.stop_gradient(pressure.values))
        velocity = velocity.with_values(math.stop_gradient(velocity.values))
        self._total_loss = math.stop_gradient(self._total_loss)
        return pressure, velocity

def get_pos_grad(input_field, center_init, multiplier, ck):
    gridSize = input_field.dx.native()[0]
    with math.record_gradients(input_field.values):
        coord = input_field.points
        mask = heaviside_atan(-1*input_field, gridSize/100)
        center_geo = math.sum((mask * coord).values, dim='x,y') / math.sum(mask.values)
        pos_diff = (center_geo - center_init) / gridSize
        pos_aug = math.l2_loss(pos_diff)
        pos_loss = math.sum(multiplier * pos_diff, dim = 'vector') + ck * pos_aug
        grad = math.gradients(pos_loss)
    pos_diff = math.stop_gradient(pos_diff)
    pos_loss = math.stop_gradient(pos_loss)
    return grad, pos_diff, pos_loss