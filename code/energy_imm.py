# # Shape optimization with phiflow
# 
# 1. Domain: pipe flow
# 2. B.C.: no slip
# 3. Initial shape: circular cylinder
# 4. Inflow flow: from left **Poiseuille flow**
# 5. Flow: unsteady laminar flow
# 6. Area constraint: shifting
# 7. Position constraint: shifting
# 8. Vortex evaluation: N/A

from phi.torch.flow import *
from phi.geom import Phi #geometry extension
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time, os, sys, logging, argparse
sys.path.append('../')
from functions import *

parser = argparse.ArgumentParser()
parser.add_argument("-re", "--Reynolds", type = int, choices = [40, 100,190], required = True, help = "set Reynolds number for simulation")
parser.add_argument("-n", "--name", type = int, required = True, help = "set file name (simulation number)")
parser.add_argument("--adam", action = 'store_true', help = "activate adam optimizer if specified")
parser.add_argument("-iter", "--iteration", type = int, default = 4000, help = "set max iteration")
parser.add_argument("-interv", "--interval", type = int, choices = [1, 10, 50, 100, 200, 500], default = 100, help = "set interval for saving intermediate results")
parser.add_argument("-res", "--resolution", type = int, default = 128, choices=[64,128,256,512], help = "set resolution")

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
    
def main():

    ################ set parameters ################
    
    args = parser.parse_args()
    re_to_velocity_time_step = {40: (0.11, 0.05), 100: (0.27, 0.02), 190: (0.5, 0.01)} #hard-coded velocity and time step given Re
    
    #set device
    TORCH.set_default_device('GPU')

    #file and data related
    sim_num = args.name
    save_interval = args.interval
    GENERATE_PLOT = True
    SAVE_INTERM_RESULT = True

    #optimization related
    res = args.resolution
    radius = 0.3
    diffusivity = 0.001
    method = 'rk2_upwind' if not args.adam else 'adam'
    vis_scale = 1. #scaling factor for viscous force
    cfl = 0.9
    convergence_tol = 1e-5
    convergence_win = 50
    re = args.Reynolds
    max_iter = args.iteration
    START_ACCU_FLAG = False
    DEBUG = True
    try:
        inflow_velocity, DT = re_to_velocity_time_step[re]
    except:
        raise ValueError("Reynolds should be 40, 100 or 190.")

    #adam related
    ACTIVATE_ADAM = args.adam
    if ACTIVATE_ADAM:
        step = 0
        lr = 1e-3
        m = 0.
        v = 0.
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 0.1
        
    ################ set logger ################

    handler = logging.FileHandler(filename = 'runtime_{}.log'.format(sim_num))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s --- %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Current device: {}".format(TORCH.get_default_device()))
    logger.info('Reynolds number set to: {}'.format(re))

    ################ set up phiflow domain ################

    #set up domain and inflow
    DOMAIN = dict(x = 2*res, y = res, bounds=Box[-1:3,-1:1], extrapolation = extrapolation.combine_sides(x = extrapolation.BOUNDARY, y = extrapolation.ZERO))
    INFLOW = StaggeredGrid(HardGeometryMask(Box[:-0.98, :]), **DOMAIN)
    velocity = StaggeredGrid(0., **DOMAIN)
    pressure = CenteredGrid(0., **DOMAIN)

    #define poiseuille inflow velocity profile
    def poiseuille_flow(field):
        x = field.staggered_direction['y'].vector['x']
        y = field.staggered_direction['x'].vector['y']
        x_values = inflow_velocity*(1 - y**2)
        y_values = 0*x
        return math.stack([x_values,y_values], channel('staggered_direction'))
    INFLOW_VELO = StaggeredGrid(poiseuille_flow, **DOMAIN)

    #for plotting
    grids = pressure.points.unstack(dimension='vector')
    x,y = [grid.numpy('y,x') for grid in grids]

    #extract info from domain and precompute the squared value
    gridSize = pressure.dx.native()[0]
    gridSizeSquared = gridSize**2

    #set up domain for phi
    DOMAIN_PHI = dict(x = 2*res, y = res, bounds=Box[-1:3,-1:1], extrapolation = extrapolation.ZERO)

    def init_phi(field):
        x,y = field.unstack(dimension = 'vector')
        return x**2 + y**2 - radius**2

    #instantiate initial phi field
    phi_field = CenteredGrid(init_phi, **DOMAIN_PHI)
    phi_geom = Phi(phi_field)
    phi_obs = Obstacle(phi_geom)
    phi_mask = HardGeometryMask(phi_geom) @ phi_field

    #regularize phi (|gradient of phi|= 1)
    phi_field = make_sdf(phi_field)

    #calculate initial area
    area_init_vector = heaviside_atan(-1*phi_field, gridSize)
    area_init = math.sum(area_init_vector.values) * gridSizeSquared
    center_init = phi_geom.center

    logger.info('Basic domain setup finished.')

    ################ load pre-stored fields ################

    with cwd('../prestored_data/res{}/dt{:03d}/poiseuille/'.format(res, int(DT*1e2))):
        with open('velocity_x_rad030_t200_vel{:04d}.txt'.format(int(inflow_velocity*1e3)), 'r') as f:
            velo_x_file = np.loadtxt(f).reshape(2*res+1, res)
        with open('velocity_y_rad030_t200_vel{:04d}.txt'.format(int(inflow_velocity*1e3)), 'r') as f:
            velo_y_file = np.loadtxt(f).reshape(2*res, res-1)
        with open('pressure_rad030_t200_vel{:04d}.txt'.format(int(inflow_velocity*1e3)), 'r') as f:
            pressure_file = np.loadtxt(f).reshape(2*res, res)

    pressure = pressure.with_values(tensor(pressure_file, pressure.shape))
    v_x = tensor(velo_x_file, velocity.vector[0].resolution)
    v_y = tensor(velo_y_file, velocity.vector[1].resolution)
    v_stack = math.stack([v_x, v_y], channel('vector'))
    velocity = velocity.with_values(v_stack)

    logger.info('Prestored data loaded.')

    ################ set up helpers ################

    accumulator = AccumulatorEnergy(phi_field, radius, diffusivity, INFLOW_VELO, INFLOW, DT, phi_obs, target = 0)

    ################ create path and optimization info ################

    #data destination
    path = '../data/'
    dir_name = 'sim{:03d}'.format(sim_num)

    try:
        os.makedirs(path + dir_name + '/img_cache')
        os.makedirs(path + dir_name + '/img_cache_mask')
        os.makedirs(path + dir_name + '/tmp')
        
        #used by Plotters
        with cwd(path + dir_name):
            abs_path = os.getcwd()
    except:
        logger.error('Data file already exists.')
        sys.exit()

    #store general infomation
    with cwd(path + dir_name):
        with open('info.txt', 'w') as f:
            f.write("""
        FLOW OPTIMIZATION WITH SHIFTING ON BARYCENTER, SHIFTING ON AREA AND IMM. UPDATE

        General info:
            resolution: {res}
            time step: {dt}
            velocity: {inflow_velo} (Poiseuille flow - centerline velocity)
            radius: {rad}
            diffusivity: {mu}
            reynolds number: {re}\n
        Technical info:
            update method: {method}
            viscosity scaling: {vis_scale}
            cfl: {cfl}
            max_iteration: {max_iter}
            convergence window: {conv_win}
            convergence tolerance: {conv_tol}""".format(
                res = res, dt = DT, inflow_velo = inflow_velocity, rad = radius, mu = diffusivity, re = re,
                method = method, vis_scale = vis_scale, cfl =cfl, max_iter = max_iter,
                conv_win = convergence_win, conv_tol = convergence_tol
        ))

    ################ prepare storage ################
            
    velocity_record = np.zeros(max_iter)
    pressure_record = np.zeros((max_iter, 2))
    viscous_record = np.zeros((max_iter, 2))
    area_record = []
    pos_record = []
    total_loss_record = []

    ################ start simulation ################

    if GENERATE_PLOT:
        #initiate plotters
        fieldPlotter = FieldPlotter(x, y, phi_field.values.numpy('y,x'), radius,
                                    velocity.at_centers().values.numpy('y,x,vector')[...,0],
                                    pressure.values.numpy('y,x'),
                                    path = abs_path + '/img_cache')
        maskPlotter = MaskPlotter(x, y, phi_field.values.numpy('y,x'), radius,
                                 phi_mask.values.numpy('y,x'),
                                 path = abs_path + '/img_cache_mask')
        #save initial plots
        fieldPlotter.savefig(0)
        maskPlotter.savefig(0)

    # run simulation
    logger.info('Optimization running......')

    #for proper pressure solve at first iteration
    pressure = None

    t_total = time.time()
    t_prog = time.time()
    
    iteration = 0
    while iteration < max_iter:

        #assign new phi
        accumulator.set_phi(phi_field)
        accumulator.set_obs()
        pressure, velocity = accumulator.run(velocity, pressure)
        pressure, velocity = accumulator.backward(velocity, pressure)
            
        #record data if backpropagation is not performed
        pressure_record[iteration,:] = accumulator.pressure_record #solve the duplication of value assignment in record
        viscous_record[iteration,:] = accumulator.viscous_record
        velocity_record[iteration] = np.mean(velocity.at_centers().values.numpy('y,x,vector')[:,10,0])
           
        grad_plus, grad_minus, t_grad = accumulator.grad_plus, accumulator.grad_minus, accumulator.grad

        #record loss
        total_loss_record.append((accumulator.total_loss).numpy())

        #check convergence
        if len(total_loss_record) > 2* convergence_win and iteration % 5 == 4:
            if check_convergence(total_loss_record, len(total_loss_record)-1, convergence_win, convergence_tol):
                break

        if ACTIVATE_ADAM:
            step += 1
            m = beta1 * m + (1 - beta1) * t_grad
            v = beta2 * v + (1 - beta2) * t_grad**2
            bias_correction1 = (1- beta1 ** step)
            bias_correction2 = (1- beta2 ** step)
            step_size = lr / bias_correction1
            t_grad = m / (math.sqrt(v/bias_correction2) + epsilon) * step_size
            phi_field = phi_field - t_grad
        else:
            dt = determine_step(t_grad, gridSize, factor = cfl)

            #update phi
            k1 = (math.where(t_grad > 0., t_grad, 0.)*grad_plus + math.where(t_grad < 0., t_grad, 0.)*grad_minus)
            phi_field_old = phi_field
            phi_field -= k1 * dt

            #apply Heun's method if true
            if method.find('rk2') != -1:
                grad_plus, grad_minus, _ = grad_upwind(phi_field)
                k2 = (math.where(t_grad > 0., t_grad, 0.)*grad_plus + math.where(t_grad < 0., t_grad, 0.)*grad_minus)
                phi_field = phi_field_old - 0.5 * (k1 + k2) * dt

        phi_field = make_sdf(phi_field)

        #impose area constraint
        shift, area, area_list, area_converged = constrain_area_adam(phi_field, area_init, rel_tolerance = 1e-6, lr = 1e-3, max_iteration = 10000)
        phi_field += CenteredGrid(shift, **DOMAIN_PHI)
        area_record.append((area - area_init)/area_init)
        if area_converged == False:
            logger.warning("Current iteration: {}".format(iteration))
            
        #impose barycenter constraint
        phi_field, corrected_pos = constrain_position(phi_field, center_init, gridSize, substeps = 15)
        pos_record.append((corrected_pos - center_init)/gridSize)

        #update geometry and zero pressure
        phi_obs = phi_obs.copied_with(geometry = Phi(phi_field))
        phi_mask = HardGeometryMask(~phi_obs.geometry) @ phi_field
        pressure = get_interpolated_pressure(pressure)
        pressure = pressure * phi_mask

        #save intermediate data
        if SAVE_INTERM_RESULT and len(total_loss_record) % save_interval == 0:
            with cwd(path + dir_name + '/tmp'):
                os.mkdir('{:05d}'.format(iteration))
            with cwd(path + dir_name + '/tmp/{:05d}'.format(iteration), print_path = True):
                interval = slice(0, iteration)
                save_data(viscous_record = viscous_record, 
                          pressure_record = pressure_record,
                          velocity_record = velocity_record,
                          velocity = velocity, 
                          pressure = pressure,
                          phi_field = phi_field, 
                          phi_grad = t_grad,
                          position_record = pos_record, 
                          area_record = area_record,
                          total_loss_record = total_loss_record,
                          interval = interval)
            logger.info('data saved.')

        #print progress
        if iteration % 10 == 9:
            logger.info("Iteration {} finished.  |  Duration: {} min".format(iteration+1, (time.time() - t_prog)/60.))
            t_prog = time.time()
            
            if GENERATE_PLOT:
                #update and save plots
                velocity_numpy = velocity.at_centers().values.numpy('y,x,vector')[...,0]
                pressure_numpy = pressure.values.numpy('y,x')
                fieldPlotter.update(velocity_numpy, pressure_numpy, phi_field.values.numpy('y,x'))
                fieldPlotter.savefig(iteration)
    
                phi_mask = HardGeometryMask(phi_obs.geometry) @ phi_field
                maskPlotter.update(phi_mask.values.numpy('y,x'), 
                                   phi_field.values.numpy('y,x'))
                maskPlotter.savefig(iteration)

        iteration += 1
        
    #save final data
    with cwd(path + dir_name + '/tmp'):
        os.mkdir('{:05d}'.format(iteration))
    with cwd(path + dir_name + '/tmp/{:05d}'.format(iteration), print_path = True):
        interval = slice(0, iteration)
        save_data(viscous_record = viscous_record, 
                    pressure_record = pressure_record,
                    velocity_record = velocity_record,
                    velocity = velocity, 
                    pressure = pressure,
                    phi_field = phi_field, 
                    phi_grad = t_grad,
                    position_record = pos_record, 
                    area_record = area_record,
                    total_loss_record = total_loss_record,
                    interval = interval)
    logger.info('data saved.')
            
    #record total iteration
    with cwd(path + dir_name):
        os.mkdir('result_img')
        with open('info.txt', 'a') as f:
            f.write("\n\tTotal iteration: {}".format(iteration))

    logger.info('optimization finished.')
    logger.info('total iterations: {}'.format(iteration))
    logger.info("Total runtime: {} hr".format((time.time() - t_total)/3600.))

if __name__ == '__main__':
    main()

