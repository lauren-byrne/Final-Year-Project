import unittest
import numpy as np


class testGazeTracking(unittest.TestCase):
    def test_npMinMax(self):
        points = np.array([(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)], np.int32)
        self.assertEqual(np.min(points[:, 0]), 2)
        self.assertEqual(np.max(points[:, 0]), 6)
        self.assertEqual(np.min(points[:, 1]), 3)
        self.assertEqual(np.max(points[:, 1]), 7)


if __name__ == '__main__':
    unittest.main()
