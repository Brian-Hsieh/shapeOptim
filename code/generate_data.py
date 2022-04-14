from phi.flow import *
from phi.geom import Phi
import matplotlib.pyplot as plt
import time, os, sys, argparse
sys.path.append('../')
from functions import *

parser = argparse.ArgumentParser()
parser.add_argument("-res", "--resolution", type = int, default = 128, choices=[64,128,256,512], help = "set resolution")
parser.add_argument("-v", "--velocity", type=float, required = True, help="set velocity at center line")
parser.add_argument("-dt", "--time_step", type=float, help="set time step")

def main():

    ################ set parameters ################

    args = parser.parse_args()
    res = args.resolution
    inflow_velocity = args.velocity
    DT = 0.5/inflow_velocity*0.01 if args.time_step == None else args.time_step
    radius = 0.3
    diffusivity = 0.001
    t_end = 10
    ep = res/128 #used for force calculation
    substeps = 20 if res == 512 else 4 #used for pressure solve

    ################ set up phiflow domain ################

    #set up domain and inflow
    DOMAIN = dict(x = 2*res, y = res, bounds=Box[-1:3,-1:1], extrapolation = extrapolation.combine_sides(x = extrapolation.BOUNDARY, y = extrapolation.ZERO))
    INFLOW = StaggeredGrid(HardGeometryMask(Box[:-0.98, :]), **DOMAIN)

    #define poiseuille inflow velocity profile
    def poiseuille_flow(field):
        x = field.staggered_direction['y'].vector['x']
        y = field.staggered_direction['x'].vector['y']
        x_values = inflow_velocity*(1 - y**2)
        y_values = 0*x
        return math.stack([x_values,y_values], channel('staggered_direction'))
    INFLOW_VELO = StaggeredGrid(poiseuille_flow, **DOMAIN)

    #set up domain for phi
    DOMAIN_PHI = dict(x = 2*res, y = res, bounds=Box[-1:3,-1:1], extrapolation = extrapolation.ZERO)

    def phi_func(field):
        x,y = field.unstack(dimension = 'vector')
        return x**2 + y**2 - radius**2

    #instantiate initial phi field
    phi_field = CenteredGrid(phi_func, **DOMAIN_PHI)
    phi_geom = Phi(phi_field)
    phi_obs = Obstacle(phi_geom)

    #regularize phi (|gradient of phi|= 1)
    phi_field = make_sdf(phi_field)

    #initialize field value
    pressure = None
    velocity = INFLOW_VELO + 10*StaggeredGrid(Noise(vector=2), **DOMAIN) * INFLOW #add noise to accelerate flow evolution

    ################ create path ################

    path = '../prestored_data/unsteady/res{res}/dt{dt:03d}/poiseuille/'.format(res=res, dt=int(DT*1e2))

    try:
        os.makedirs(path)
    except:
        print('Data file already exists.')
        sys.exit()

    ################ prepare storage ################

    pressure_record = np.zeros((int(t_end/DT),2))
    viscous_record = np.zeros((int(t_end/DT),2))
    velocity_record = np.zeros(int(t_end/DT))

    ################ start simulation ################

    t_start = time.time()
    for i, t in enumerate(np.arange(0, t_end, DT)):

        velocity = advect.semi_lagrangian(velocity, velocity, DT)
        velocity = velocity * (1- INFLOW) + INFLOW * INFLOW_VELO
        velocity = diffuse.explicit(velocity, diffusivity, DT, substeps = substeps)
        velocity, pressure = fluid.make_incompressible(velocity, 
                                                        obstacles = (phi_obs,), 
                                                        solve=math.Solve('auto', 1e-3, 0, x0 = pressure, max_iterations=1e4, gradient_solve=math.Solve('auto', 1e-5, 1e-5)))
        velocity_record[i] = np.mean(velocity.at_centers().values.numpy('y,x,vector')[:,10,0])

        pressure_force, viscous_force = evaluate_force(phi_field, pressure/DT, velocity, diffusivity, epsilon_factor = ep)
        pressure_record[i,:] = pressure_force
        viscous_record[i,:] = viscous_force

        if i % 100 == 0:
            print('Iteration {} finished --- time spent: {}min'.format(i, (time.time() - t_start)/60))
            t_start = time.time()

    with cwd(path):
        with open('velocity_x_rad030_t200_vel{:04d}.txt'.format(int(inflow_velocity*1e3)), 'w') as f:
            for elem in velocity.vector[0].values.numpy('x,y'):
                np.savetxt(f, elem)
        with open('velocity_y_rad030_t200_vel{:04d}.txt'.format(int(inflow_velocity*1e3)), 'w') as f:
            for elem in velocity.vector[1].values.numpy('x,y'):
                np.savetxt(f, elem)
        with open('pressure_rad030_t200_vel{:04d}.txt'.format(int(inflow_velocity*1e3)), 'w') as f:
            for elem in pressure.values.numpy('x,y'):
                np.savetxt(f, elem)

        with open('velocity_record.txt', 'w') as f:
            np.savetxt(f, velocity_record)
        with open('pressure_record.txt', 'w') as f:
            np.savetxt(f, pressure_record)
        with open('viscous_record.txt', 'w') as f:
            np.savetxt(f, viscous_record)

        plt.figure()
        plt.plot(pressure_record[:,0])
        plt.grid()
        plt.savefig('pressure drag evolution')

if __name__ == '__main__':
    main()