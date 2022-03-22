from phi.math import tensor
from phi.field import Field, CenteredGrid
import skfmm

def make_sdf(field: CenteredGrid):
    field_numpy = field.values.numpy('x,y')
    field_sdf = skfmm.distance(field_numpy, dx = field.dx.native())
    return field.with_values(tensor(field_sdf, field.shape))