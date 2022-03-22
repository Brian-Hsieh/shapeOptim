from phi import math
from phi.field import HardGeometryMask
from ._geom import Geometry, _fill_spatial_with_singleton
from ..math import wrap

class Phi(Geometry):
    def __init__(self, field):
        #the input field is assumed to be sdf
        self._field = field
        self._center = wrap([0.] * self.field.shape.rank)

    @property
    def center(self):
        coord = self.field.points
        mask = HardGeometryMask(Phi(self.field)) @ self.field
        center_geo = math.sum((mask * coord).values, dim='x,y') / math.sum(mask.values)
        return center_geo
    
    @property
    def shape(self):
        return _fill_spatial_with_singleton(self._center.shape)
    
    @property
    def field(self):
        return self._field

    def bounding_radius(self):
        return math.abs(self.approximate_signed_distance(self.center))
    
    def lies_inside(self, location):
        local_points = self.field.box.global_to_local(location) * self.field.resolution - 0.5
        t = math.grid_sample(self.field.values, local_points, self.field.extrapolation)
        return t <= 0

    def approximate_signed_distance(self, location):
        local_points = self.field.box.global_to_local(location) * self.field.resolution - 0.5
        t = math.grid_sample(self.field.values, local_points, self.field.extrapolation)
        return t

    def set_field(self, new_field):
        self._field = new_field