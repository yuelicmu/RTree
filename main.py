# class R-Tree
from Shapes import Rectangle, Circle, Line, Point, BoundingBox
import numpy as np
import random

class RTree:
    def __init__(self, M=5):
        self.M = M
        self.head = None
        self.size = 0

    def insert(self, E):
        """
        Insert a new object E into self.
        :param self: an RTree object
        :param E: an object of shape classes
        """
        self.size += 1
        if not self.head:
            self.head = LeafNode([E])
        else:
            L = self.choose_leaf(E)
            if len(L.elem) < self.M:
                L.elem.append(E)
                L.update_bounding_box()
                self.adjust_bound(L)
            else:
                [L, LL] = RTree.split_node(L,E)
                L.update_bounding_box()
                LL.update_bounding_box()
                self.adjust_tree(L, LL)
                self.adjust_bound(L, LL)

    def choose_leaf(self, E):
        """
        Choose a leaf node in self to insert E.
        :param self: an RTree object
        :param E: an object of shape classes
        :return a leaf node in self
        """
        current_tree = self.head
        while True:
            if isinstance(current_tree, LeafNode):
                return current_tree
            else:
                F = RTree.min_increase(current_tree, E)
                current_tree = F

    @staticmethod
    def min_increase(node, E):
        """
        Given a node and a new index entry E, find the children of it that has the minimal
        increase in area when adding E.
        :param E: an object of shape classes
        :param node: a non-leaf node with children
        :return a children of node
        """
        if not isinstance(node, Node):
            raise ValueError('Wrong input: leaf node.')
        else:
            bb_2 = E.bounding_box
            F = node.children[0]
            min_inc = F.bounding_box.include(bb_2).area - F.bounding_box.area
            for item in node.children[1:]:
                increase = item.bounding_box.include(bb_2).area - item.bounding_box.area
                if increase < min_inc:
                    F, min_inc = item, increase
        return F

    @staticmethod
    def add_bb(node, new_bb):
        node.bounding_box = node.bounding_box.add_bb(new_bb)

    def adjust_tree(self, L, LL):
        N, NN = L, LL
        while True:
            if not N.father:
                new_head = Node([N, NN])
                self.head = new_head
                return
            else:
                P = N.father
                if len(P.children) < self.M-1:
                    P.children.append(L)
                    P.children.append(LL)
                    return
                else:
                    P.children.append(L) # needs modification
                    N, NN = self.split_node(P,LL)


    def adjust_bound(self, L, LL=None):
        if LL:
            bb = all(bb(L), bb(LL))
        else:
            bb = bb(L)
        father = L.father
        while True:
            if not father:
                return
            else:
                adjust_father_bb(bb)
                bb = father.bb()
                father = father.father


    @staticmethod
    def split_node(L, E):
        """
        Split function, static method
        :param L: a leaf node of a Rtree
        :param E: a shape object to be inserted into L
        :return: two leaf nodes with random split(a list)
        """
        father = L.father
        l_new = L.elem.copy()
        l_new.append(E)
        random.shuffle(l_new)
        ll = int(len(l_new)/2)
        left_child, right_child= l_new[0:ll], l_new[ll:]
        left_leaf, right_leaf = LeafNode(left_child), LeafNode(right_child)
        left_leaf.father, right_leaf.father = father, father
        return [left_leaf, right_leaf]


class LeafNode:
    """
    Class for leaf node.
    init: leaf = LeafNode()
    """

    def __init__(self, L=[]):
        self.elem = L
        self.father = None
        self.bounding_box = None

    def update_bounding_box(self):
        return(1)


class Node:
    """
    Class for non-leaf node.
    init: node = Node()
    """

    def __init__(self, L=[]):
        self.children = L
        self.father = None
        self.bounding_box = None

    def update_bounding_box(self):
        return(1)

if __name__=='__main__':
    circ_1 = Circle(np.array([1,2]), 1)
    rtree = RTree(5)
    rtree.insert(circ_1)
    print(rtree.head.elem)
    leaf_1 = LeafNode([circ_1])

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
        if not self.shape_1.bounding_box or not self.shape_2.bounding_box:
            return None
        bb_1 = self.shape_1.bounding_box
        bb_2 = self.shape_2.bounding_box
        if not bb_1.intersection(bb_2):
            return None
        else:
            x1, x2 = max(bb_1.xlower, bb_2.xlower), min(bb_1.xupper, bb_2.xupper)
            y1, y2 = max(bb_1.ylower, bb_2.ylower), min(bb_1.yupper, bb_2.yupper)
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
        self.xlower = min(self.xlower, new_bb.xlower)
        self.ylower = min(self.ylower, new_bb.ylower)
        self.xupper = max(self.xupper, new_bb.xupper)
        self.yupper = max(self.yupper, new_bb.yupper)

    def include(self, new_bb):
        return (BoundingBox(min(self.xlower, new_bb.xlower),
                            max(self.xupper, new_bb.xupper),
                            min(self.ylower, new_bb.ylower),
                            max(self.yupper, new_bb.yupper)))

import unittest
import numpy as np
from Shapes import Rectangle, Circle, Line, Point, BoundingBox, ShapeUnion
from RTree import RTree, Node, LeafNode


class testRTee(unittest.TestCase):
    # create a case for testing
    def generate_case(self):
        node_lst = []
        for i in range(50):
            circ = Circle(np.array([i, i]), 1)
            node_lst.append(circ)
        new_obj = Circle(np.array([10, 10]), 1)
        return [node_lst, new_obj]

    # test construction function
    def test_insert(self):
        node_lst = self.generate_case()[0]
        r_tree = RTree()
        for item in node_lst:
            r_tree.insert(item)
        self.assertEqual(r_tree.size, 50)
        self.assertTrue(isinstance(r_tree.head, Node))
        return r_tree

    def test_minimal_increase(self):
        r_tree = self.test_insert()
        node = r_tree.head
        pass

    def test_choose_leaf(self):
        r_tree = self.test_insert()
        new_obj = self.generate_case()[1]
        leaf = r_tree.choose_leaf(new_obj)
        self.assertTrue(isinstance(leaf, LeafNode))

    def test_add_bb(self):
        r_tree = self.test_insert()
        new_obj = self.generate_case()[1]
        bb_1 = r_tree.head.bouding_box
        self.assertEqual(bb_1, 0)

    def test_adjust_tree(self):
        pass

    def test_split_node(self):
        new_obj = self.generate_case()[1]
        node_lst = []
        for i in range(5):
            circ = Circle(np.array([i, i]), 1)
            node_lst.append(circ)
        L = LeafNode(node_lst)
        [left_leaf, right_leaf] = RTree.split_node(L, new_obj)
        self.assertTrue(isinstance(left_leaf, LeafNode))
        self.assertTrue(isinstance(right_leaf, LeafNode))


if __name__ == '__main__':
    unittest.main()

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

from Shapes import Rectangle, Circle, Line, Point, BoundingBox
from RTree import RTree
import numpy as np
import csv

# construct query list and crime tree
query_lst = []
with open('crime-boxes.csv', newline='') as csvfile:
    box_reader = list(csv.reader(csvfile, delimiter=','))
    dim_name_query = box_reader[0]
    query_names = []
    for i in range(1, len(box_reader)):
        query_names.append(box_reader[i][0])
        new_line = box_reader[i][1:]
        x1, y1, x2, y2 = list(map(float,new_line))
        new_rec = Rectangle(np.array([x1,y1]), np.array([x2,y2]), angle=0, id=None)
        query_lst.append(new_rec)

crime_tree = RTree()
with open('crimes.csv', newline='') as csvfile:
    crime_reader = list(csv.reader(csvfile, delimiter=','))
    dim_name_crime = crime_reader[0]
    for i in range(1, len(crime_reader)):
        new_line = crime_reader[i]
        new_rec = Rectangle(np.array([0,0]), np.array([1,1]), angle=0, id=None)
        # crime_tree.insert(new_rec)

# analysis of crime data in query list
# The numbers of robberies and assaults within that region

# The most recent crime of each type within that region

# The density of crimes per unit area
if __name__=='__main__':
    print(query_lst)
    print(query_names)

# This is a command-line driver script to locate all the crimes within a user-supplied box.
# example:
# python find-crimes.py --section 2702 414970 123799 417057 126342
# will print all the aggravated assaults within the box defined by those coordinates.
import argparse
from Shapes import Rectangle, Circle, Line, Point, BoundingBox, ShapeUnion
from RTree import RTree
import numpy as np
from data_analysis import crime_tree

parser = argparse.ArgumentParser()
parser.add_argument('-X', '--Xdata', help='n*p data matrix to be loaded where n is sample size.',
                    type=str)
args = parser.parse_args()
Tree = RTree(eval(args.Xdata),leafsize=args.leafsize)
