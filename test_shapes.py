import unittest
import numpy as np
from Shapes import Rectangle, Circle, Line, Point, BoundingBox, ShapeUnion


class testRTee(unittest.TestCase):
    def test_Rectangle(self):
        rec = Rectangle(np.array([1, 1]), np.array([2, 2]), np.pi / 4)
        rec_bb = rec.bounding_box
        rec_bb_1, rec_bb_4 = rec_bb.xlower, rec_bb.yupper
        self.assertTrue(np.abs(rec_bb_1 - 1 + np.sqrt(2) / 2) <= 10 ** -6)
        self.assertTrue(np.abs(rec_bb_4 - 1 - np.sqrt(2)) <= 10 ** -6)

    def test_Point(self):
        point_bb = Point(np.array([1, 1])).bounding_box
        self.assertEqual(point_bb.xlower, 1)
        self.assertEqual(point_bb.yupper, 1)

    def test_Line(self):
        line_bb = Line(np.array([0, 0]), np.array([1, 1])).bounding_box
        self.assertEqual(line_bb.xlower, 0)
        self.assertEqual(line_bb.yupper, 1)

    def test_Circle(self):
        circ_bb = Circle(np.array([1, 1]), 1).bounding_box
        self.assertEqual(circ_bb.xlower, 0)
        self.assertEqual(circ_bb.yupper, 2)

    def test_insertction(self):
        bb1 = BoundingBox(0, 1, 0, 1)
        bb2 = BoundingBox(1 / 2, 3, 0, 4)
        self.assertTrue(bb1.intersection(bb2))

    def test_contain(self):
        bb1 = BoundingBox(0, 1, 0, 1)
        bb2 = BoundingBox(1 / 2, 1, 0, 1)
        self.assertTrue(bb1.contain(bb2))

    def test_add_bb(self):
        bb1 = BoundingBox(0, 1, 0, 1)
        bb2 = BoundingBox(1 / 2, 3, 0, 4)
        bb1.add_bb(bb2)
        self.assertEqual(bb1.xlower, 0)
        self.assertEqual(bb1.yupper, 4)

    def test_shape_union(self):
        shape_1 = Circle(np.array([1, 1]), 1)
        shape_2 = Line(np.array([-2, -2]), np.array([-1, -1]))
        shape_union = ShapeUnion(shape_1, shape_2)
        self.assertEqual(shape_union.bounding_box, None)


if __name__ == '__main__':
    unittest.main()
