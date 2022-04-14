#!./.venv/Scripts/python.exe

import unittest
import numpy as np
from open3d import utility as o3d_util
from open3d import geometry as o3d_geometry
from hyperviz.models import O3DBaseModel, O3DPointCloudModel, O3DTriangleMeshModel

class TestModels(unittest.TestCase):
    
    def test_o3d_typechecks(self):
        # base model cannot be called directly
        with self.assertRaises(TypeError):
            bm = O3DBaseModel()
        # can instantiate models
        pointcloud_model = O3DPointCloudModel("pntcld")
        stl_model = O3DTriangleMeshModel("stl")

        # wrong types raise TypeErrors
        with self.assertRaises(TypeError):
            O3DTriangleMeshModel('should_fail', o3d_geometry.PointCloud())
        with self.assertRaises(TypeError):
            O3DPointCloudModel('should_fail', o3d_geometry.TriangleMesh())

    def test_asO3Ddict(self):
        pcm = O3DPointCloudModel('pntcld')
        # should not raise an error
        pcm.asO3Ddict()['name']
        # should raise an error
        with self.assertRaises(KeyError):
            pcm.asO3Ddict()['material']
        # should not raise an error after attribute change
        pcm.transparent = True
        pcm.asO3Ddict()['material']

    def test_cartisian_transform(self):
        pc = O3DPointCloudModel('pointcloud')
        points = o3d_util.Vector3dVector(np.array([[0, 0, 0]]))
        pc.points = points
        position_vector = np.asarray([5, 3, 2])
        rotation_vector = np.array([np.pi, np.pi, np.pi])/2
        # move to 5, 3, 2
        pc.geometry.cartisian_transform(position_vector, rotation_vector)
        self.assertTrue(np.array_equal(position_vector, pc.cartisian_coordinates[0]))
        self.assertTrue(np.array_equal(rotation_vector, pc.cartisian_coordinates[1]))
    


if __name__ == '__main__':
    unittest.main(verbosity=2)