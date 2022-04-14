import unittest
import numpy as np
from hyperviz.linear_color_mapper import LinearColorMapper, LinearColorMapperItem


class TestLinearColorMapper(unittest.TestCase):

    def test_add_item(self):
        mapper = LinearColorMapper()
        item = LinearColorMapperItem(
                np.array([0, 0, 0], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.float64),
                0, 1
            )
        mapper.add_item(item)

    def test_remove_item(self):
        mapper = LinearColorMapper()
        item = LinearColorMapperItem(
                np.array([0, 0, 0], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.float64),
                0, 1
            )
        mapper.add_item(item)
        mapper.remove_item(item)

    def test_get_color(self):
        mapper = LinearColorMapper()
        item = LinearColorMapperItem(
                np.array([0, 0, 0], dtype=np.float64),
                np.array([1, 1, 1], dtype=np.float64),
                0, 1
            )
        mapper.add_item(item)
        color = mapper.get_color(0)
        


if __name__ == '__main__':
    unittest.main(verbosity=2)