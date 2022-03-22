
import numpy as np

def save_data(viscous_record, pressure_record,
              velocity_record,
              velocity, pressure,
              phi_field, phi_grad,
              position_record, area_record,
              total_loss_record,
              interval,
              multiplier_position_record = None,
              loss_record = None,
              position_update_list = None):

    with open('viscous_record', 'w') as f:
        for elem in viscous_record[interval, :]:
            np.savetxt(f, elem)
    with open('pressure_record', 'w') as f:
        for elem in pressure_record[interval, :]:
            np.savetxt(f, elem)

    with open('velocity_record', 'w') as f:
        np.savetxt(f, velocity_record)

    with open('velocity_field', 'w') as f:
        for elem in velocity.at_centers().values.numpy('y,x,vector'):
            np.savetxt(f, elem)
    with open('pressure_field', 'w') as f:
        for elem in pressure.values.numpy('y,x,vector'):
            np.savetxt(f, elem)
    with open('velocity_x', 'w') as f:
        for elem in velocity.vector[0].values.numpy('y,x'):
            np.savetxt(f, elem)
    with open('velocity_y', 'w') as f:
        for elem in velocity.vector[1].values.numpy('y,x'):
            np.savetxt(f, elem)
    with open('phi_field', 'w') as f:
        for elem in phi_field.values.numpy('y,x,vector'):
            np.savetxt(f, elem)
    with open('phi_grad', 'w') as f:
        for elem in phi_grad.numpy('y,x,vector'):
            np.savetxt(f, elem)
    with open('position_record', 'w') as f:
        for elem in position_record:
            np.savetxt(f, elem)
    with open('area_record', 'w') as f:
        np.savetxt(f, area_record)
    with open('total_loss', 'w') as f:
        np.savetxt(f, total_loss_record)

    if multiplier_position_record is not None:
        with open('multiplier_pos_record', 'w') as f:
            for elem in multiplier_position_record:
                np.savetxt(f, elem)
    if loss_record is not None:
        with open('loss', 'w') as f:
            np.savetxt(f, loss_record)
    if position_update_list is not None:
        with open('pos_update_record', 'w') as f:
            np.savetxt(f, position_update_list)