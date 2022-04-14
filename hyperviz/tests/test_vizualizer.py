import unittest
from open3d import geometry as o3d_geometry
from hyperviz.vizualizer import Vizualizer
from hyperviz.models import O3DPointCloudModel


class TestVizualizer(unittest.TestCase):

    def test_can_add_and_remove_children(self):
        viz = Vizualizer()
        pcm = O3DPointCloudModel('test_cld')
        viz.add_model(pcm)
        self.assertEqual(len(viz.model_tup), 1)
        viz.remove_model(pcm)
        self.assertEqual(len(viz.model_tup), 0)

    def test_raises_error_for_duplicates(self):
        viz = Vizualizer()
        pcm = O3DPointCloudModel('test_cld')
        viz.add_model(pcm)
        # object to object comparison
        with self.assertRaises(ValueError):
            viz.add_model(pcm)
        # name to name comparison
        with self.assertRaises(ValueError):
            viz.validate_model_name(pcm)

    def test_can_detect_update_requests(self):
        viz = Vizualizer()
        pcm = O3DPointCloudModel('test_cld')
        viz.add_model(pcm)
        self.assertTrue(pcm in [m for m in viz.model_tup if m.needs_update()])
    
if __name__ == '__main__':
    unittest.main(verbosity=2)