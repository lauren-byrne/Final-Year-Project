import unittest
from pointMaths import midpoint
from pointMaths import distance


class Testmidpoint(unittest.TestCase):
    def test_distance(self):
        p1 = (-3, 4)
        p2 = (5, 4)

        p3 = (-3, 2)
        p4 = (5, 8)
        self.assertEqual(distance(p1, p2), 8)
        self.assertEqual(distance(p3, p4), 10)


if __name__ == '__main__':
    unittest.main()
