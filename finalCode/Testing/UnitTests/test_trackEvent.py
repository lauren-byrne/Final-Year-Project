import unittest
import trackEvent


class TestcropImage(unittest.TestCase):
    def test_contour_count(self):
        approx = [2, 3, 4, 5]
        approx2 = [3, 5, 4, 2, 1]
        approx3 = [2, 4, 1]
        self.assertTrue((len(approx) == 4), True)
        self.assertFalse((len(approx2) == 4), False)
        self.assertFalse((len(approx3) == 4), False)

if __name__ == '__main__':
    unittest.main()
