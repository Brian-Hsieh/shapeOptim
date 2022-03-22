from phi import field, math
from phi.field import CenteredGrid
import torch

# def heaviside(input_field: CenteredGrid, epsilon=1):
#     return 1/(1+field.exp(-1*input_field / epsilon))

# def heaviside_tanh(input_field: CenteredGrid, epsilon=1):
#     torch_tensor = 0.5 * (1+torch.tanh(input_field.values.native() / epsilon))
#     return input_field.with_(values = math.tensor(torch_tensor, 'x,y'))

def heaviside_atan(input_field: CenteredGrid, epsilon=1):
    torch_tensor = 0.5 + 1/math.PI * torch.atan(input_field.values.native('x,y') / epsilon)
    return input_field.with_values(math.tensor(torch_tensor, input_field.shape))   

def dirac(x, k=1):
    return 1/math.PI * k/(k**2*x**2+1)