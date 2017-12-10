# shape classes in two-dimension
import numpy as np


class Rectangle:
    """
    Any rectangles, can be trivial(segment of line).
    @:param: Specify any three vertexes of the rectangle as numpy array
    @:param: id should be a list of all the entries with names
    """

    def __init__(self, lower_left, upper_right, angle, id=None):
        self.lower_left = lower_left
        self.upper_right = upper_right
        self.lower_right = np.array([self.upper_right[0], self.lower_left[1]])
        self.upper_left = np.array([self.lower_left[0], self.upper_right[1]])
        self.angle = angle
        self.bounding_box = self.get_bounding_box()
        self.is_leaf = True
        self.id = id

    def get_bounding_box(self):
        lower_right_new = self.rotate(self.lower_right, self.lower_left, self.angle)
        upper_left_new = self.rotate(self.upper_left, self.lower_left, self.angle)
        upper_right_new = self.rotate(self.upper_right, self.lower_left, self.angle)
        lower_left_new = self.lower_left
        L = [lower_right_new, upper_left_new, upper_right_new, lower_left_new]
        x2, y2 = np.amax(L, axis=0)
        x1, y1 = np.min(L, axis=0)
        return BoundingBox(x1, x2, y1, y2)

    @staticmethod
    def rotate(point, center, theta):
        x, y = point[0] - center[0], point[1] - center[1]
        x_new = np.cos(theta) * x - np.sin(theta) * y
        y_new = np.sin(theta) * x + np.cos(theta) * y
        return np.array([x_new, y_new]) + center


class Point:
    """
    Point class in R2
    @:param: the point as numpy array
    """

    def __init__(self, A, id=None):
        self.point = A
        self.bounding_box = BoundingBox(A[0], A[0], A[1], A[1])
        self.is_leaf = True
        self.id = id


class Line:
    """
    Line segment class in R2.
    @:param: the two vertexes of the segment.
    """

    def __init__(self, start, end, id=None):
        self.start = start
        self.end = end
        self.bounding_box = self.get_bounding_box()
        self.is_leaf = True
        self.id = id

    def get_bounding_box(self):
        x1, x2 = sorted([self.start[0], self.end[0]])
        y1, y2 = sorted([self.start[1], self.end[1]])
        return BoundingBox(x1, x2, y1, y2)


class Circle:
    """
    Circle class in R2.
    @:param: the center and radius. Radius should be non-negative.
    """

    def __init__(self, center, r, id=None):
        self.center = center
        self.r = r
        self.bounding_box = self.get_bounding_box()
        self.id = id

    def get_bounding_box(self):
        x1, x2 = self.center[0] - self.r, self.center[0] + self.r
        y1, y2 = self.center[1] - self.r, self.center[1] + self.r
        return BoundingBox(x1, x2, y1, y2)


class ShapeUnion:
    """
    Union-of-shape class in R2.
    @:param: any number of Rectangle, Point, Line, Circle or ShapeUnion class.
    Note that this ShapseUnion class does not specify the tightest bound.
    """

    def __init__(self, shape_1, shape_2, id=None):
        self.shape_1 = shape_1
        self.shape_2 = shape_2
        self.bounding_box = self.get_bounding_box()
        self.is_leaf = True
        self.id = id
        if not self.bounding_box:
            self.shape_1 = None
            self.shape_2 = None

    def get_bounding_box(self):
        bb_1 = self.shape_1.bounding_box
        bb_2 = self.shape_2.bounding_box
        x1, x2 = min(bb_1.xlower, bb_2.xlower), max(bb_1.xupper, bb_2.xupper)
        y1, y2 = min(bb_1.ylower, bb_2.ylower), max(bb_1.yupper, bb_2.yupper)
        return BoundingBox(x1, x2, y1, y2)


class BoundingBox:
    """
    A class for bounding box of all the shape classes.
    @:param: the four axes for bounding box.
    """

    def __init__(self, xlower, xupper, ylower, yupper):
        if xlower > xupper or ylower > yupper:
            raise ValueError('Illegal input: lower bound is larger then upper bound.')
        self.xlower = xlower
        self.xupper = xupper
        self.ylower = ylower
        self.yupper = yupper
        self.area = (xupper - xlower) * (yupper - ylower)

    def intersection(self, bb_2):
        return (self.xupper >= bb_2.xlower and bb_2.xupper >= self.xlower
                and self.yupper >= bb_2.ylower and bb_2.yupper >= self.ylower)

    def contain(self, bb_2):
        return (bb_2.xlower >= self.xlower and bb_2.xupper <= self.xupper
                and bb_2.ylower >= self.ylower and bb_2.yupper <= self.yupper)

    def add_bb(self, new_bb):
        return BoundingBox(min(self.xlower, new_bb.xlower), max(self.xupper, new_bb.xupper),
                           min(self.ylower, new_bb.ylower), max(self.yupper, new_bb.yupper))

    def print(self):
        print([self.xlower, self.xupper, self.ylower, self.yupper])

    def copy(self):
        return BoundingBox(self.xlower,self.xupper,self.ylower,self.yupper)


