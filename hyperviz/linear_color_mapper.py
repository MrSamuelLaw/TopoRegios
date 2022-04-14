import numpy as np
from typing import List



class LinearColorMapperItem():
    """Maps rgb values between two colors"""

    def __init__(self, lower_rgb: np.array, upper_rgb: np.array, lower_val: float, upper_val: float):
        # assign the values
        self._lower_rgb = lower_rgb
        self._upper_rgb = upper_rgb
        self._lower_val = lower_val
        self._upper_val = upper_val

        # determine the unit vector 
        rgb_vec = (self._upper_rgb - self._lower_rgb)
        self._rgb_magnitude = np.linalg.norm(rgb_vec)
        self._rgb_unit_vec = rgb_vec/self._rgb_magnitude

        # create the color map
        self._slope, self._intercept, *_ = np.polyfit([lower_val, upper_val], [0, self._rgb_magnitude], 1)

    def in_bounds(self, value: float):
        """Returns if the color is valid for a given color map"""
        return not (self._upper_val < value < self._lower_val)

    def get_color(self, value: float):
        """Computes the rbg color for a given value"""
        if self.in_bounds(value):
            step_size = (self._slope*value + self._intercept)
            color = self._lower_rgb + (self._rgb_unit_vec*step_size)
            return color

    def recompute(self, lower_val: float, upper_val: float):
        # recompute the linear mapping of the bounds
        self._lower_val = lower_val
        self._upper_val = upper_val
        self._slope, self._intercept, *_ = np.polyfit([lower_val, upper_val], [0, self._rgb_magnitude], 1)



class LinearColorMapper():
    """Class that contains creates linear mappings between colors and values.
    Uses basic vector calculus to return results quickly"""

    def __init__(self, color_map_list: List[LinearColorMapperItem] = []):
        self.color_map_list = [*color_map_list]
        self.update_bounds()

    def update_bounds(self):
        if self.color_map_list:
            self.color_map_list.sort(key=lambda x: x._lower_val)
            self._lower_val = min([cmap._lower_val for cmap in self.color_map_list])
            self._upper_val = max([cmap._lower_val for cmap in self.color_map_list])
        else:
            self._lower_val = None
            self._upper_val = None

    def add_item(self, item: LinearColorMapperItem):
        self.color_map_list.append(item)
        self.update_bounds()

    def create_item(self, lower_rgb: np.array, upper_rgb: np.array, lower_val: float, upper_val: float):
        item = LinearColorMapperItem(lower_rgb, upper_rgb, lower_val, upper_val)
        self.add_item(item)

    def remove_item(self, item):
        self.color_map_list = [cmap for cmap in self.color_map_list if cmap is not item]
        self.update_bounds()

    def in_bounds(self, value: float):
        return not (self._upper_val < value < self._lower_val)

    def get_color(self, value: float):
        if self.in_bounds(value):
            map_item = [cmap for cmap in self.color_map_list if cmap.in_bounds(value)][0]
            return map_item.get_color(value)