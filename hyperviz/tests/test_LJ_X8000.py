import unittest
import numpy as np
import LJ_X8000
from PIL import Image
from pathlib import Path

BMP_PATH = Path(r'./path/to/height_map.bmp')

class Test_LJ_X8000(unittest.TestCase):

    def test_read_bmp(self):
        data = np.array(Image.open(str(BMP_PATH)))
        data = LJ_X8000.bmp_to_height_map(data)
        rows, cols = data.shape
        self.assertEqual(rows, 11_000)
        self.assertEqual(cols, 3_200)

    def test_height_map_to_point_cloud(self):
        data = np.array(Image.open(str(BMP_PATH)))
        data = LJ_X8000.bmp_to_height_map(data)
        data = LJ_X8000.height_map_to_np_array(data)
        self.assertTrue(data.points)

    def test_bmp_to_point_cloud(self):
        point_cloud = LJ_X8000.bmp_to_point_cloud(BMP_PATH)
        self.assertTrue(point_cloud.points)

if __name__ == '__main__':
    unittest.main(verbosity=2)
