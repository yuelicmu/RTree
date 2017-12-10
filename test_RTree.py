import unittest
import numpy as np
from Shapes import Rectangle, Circle, Line, Point, BoundingBox, ShapeUnion
from RTree import RTree, Node, LeafNode


class testRTee(unittest.TestCase):
    # create a case for testing
    def generate_case(self):
        node_lst = []
        for i in range(6):
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
        self.assertEqual(r_tree.size, 6)
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
        bb_1 = r_tree.head.bounding_box
        print(bb_1.xlower, bb_1.xupper, bb_1.ylower, bb_1.yupper)
        self.assertEqual(bb_1.area, 4)

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
