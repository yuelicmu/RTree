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
                RTree.add_child(L, E)
                self.adjust_bound(L)
            elif not L.father:
                [L, LL] = RTree.split_head(L, E)
                new_head = Node([L, LL])
                self.head = new_head
            else:
                [L, LL] = RTree.split_node(L, E)
                P = self.adjust_tree(L, LL)
                self.adjust_bound(P)

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
            min_inc = F.bounding_box.add_bb(bb_2).area - F.bounding_box.area
            for item in node.children[1:]:
                increase = item.bounding_box.add_bb(bb_2).area - item.bounding_box.area
                if increase < min_inc:
                    F, min_inc = item, increase
        return F

    @staticmethod
    def add_bb(node, new_bb):
        if not node.bounding_box:
            node.bounding_box = new_bb.copy()
        else:
            node.bounding_box = node.bounding_box.add_bb(new_bb)

    @staticmethod
    def add_child(node, child):
        node.add_child(child)

    def adjust_tree(self, L, LL):
        """
        Adjust the tree structure
        :param L: one node of r-tree, either leaf or non-leaf, not head node
        :param LL: one node of r-tree, either leaf or non-leaf, same father as L
        :return: adjusted tree (without adjusting bounding box)
        """
        N, NN, P = L, LL, L.father
        while len(P.children) > self.M:
            children_lst = P.children
            random.shuffle(children_lst)
            nn = int(len(children_lst) / 2)
            left_lst, right_lst = children_lst[0:nn], children_lst[nn:]
            N, NN = Node(left_lst), Node(right_lst)
            if P.father:
                P.father.add_child(N)
                P.father.add_child(NN)
                P.father.delete_child(P)
            else:
                self.head = Node([N, NN])
                break
            P = N.father
        return P

    def adjust_bound(self, L):
        """
        Update the bounding box of each node
        :param L: a leaf node in RTree
        """
        bb, f = L.bounding_box, L.father
        while f:
            RTree.add_bb(f, bb)
            bb = f.bounding_box
            f = f.father

    @staticmethod
    def split_node(L, E):
        """
        Split function, static method
        :param L: a leaf node of a Rtree, and not the tree head
        :param E: a shape object to be inserted into L
        :return: two leaf nodes with random split(a list)
        """
        l_new = L.elem + [E]
        random.shuffle(l_new)
        ll = int(len(l_new) / 2)
        left_child, right_child = l_new[0:ll], l_new[ll:]
        left_leaf, right_leaf = LeafNode(left_child), LeafNode(right_child)
        L.father.add_child(left_leaf)
        L.father.add_child(right_leaf)
        L.father.delete_child(L)
        return [left_leaf, right_leaf]

    @staticmethod
    def split_head(L, E):
        """
        Split function, static method. Used when L is the head node.
        :param L: a leaf node of a Rtree, the tree head
        :param E: a shape object to be inserted into L
        :return: two leaf nodes with random split(a list)
        """
        l_new = L.elem + [E]
        random.shuffle(l_new)
        ll = int(len(l_new) / 2)
        left_child, right_child = l_new[0:ll], l_new[ll:]
        left_leaf, right_leaf = LeafNode(left_child), LeafNode(right_child)
        return [left_leaf, right_leaf]

    def delete(self, E):
        """
        Delete an element E from the tree.
        :param E: a shape object. E should be contained in one leaf of the RTree
        :return: operation on the original tree. If not found, return None.
        """
        lef = self.find_leaf(E)
        lef.delete_child(E)
        self.condense_tree(lef)

    @staticmethod
    def find_leaf(T, E):
        """
        Find the leaf node containing E.
        :param T: a leaf or non-leaf node type object.
        :param E: a shape object. E should be contained in one leaf of the RTree
        :return: a leaf node containing E. If not found, return None.
        """
        bb_0 = E.bounding_box
        if isinstance(T, LeafNode):
            if E not in T.elem:
                return None
            else:
                return T
        elif isinstance(T, Node):
            for child in T.children:
                if child.bounding_box.intersection(bb_0):
                    F = RTree.find_leaf(child, E)
                    if F:
                        return F
            return None

    def insert_node(self, N):
        """
        Insert a node into the RTree, such that the leaves are on the same level.
        :param N: a Leaf node or non-leaf node object.
        """
        if not self.head:
            self.head = N
        else:
            L = self.choose_node(N)
            if L.num < self.M:
                RTree.add_child(L, N)
                self.adjust_bound(L)
            elif not L.father:
                [L, LL] = RTree.split_head(L, E)
                new_head = Node([L, LL])
                self.head = new_head
            else:
                [L, LL] = RTree.split_node(L, E)
                P = self.adjust_tree(L, LL)
                self.adjust_bound(P)

    def choose_node(self, N):
        """
        Find the position to insert N, a Leaf or non-leaf object.
        :param N: a Leaf node or non-leaf node object.
        """
        current_tree = self.head
        if isinstance(N, Node):
            order = N.order
        else:
            order = 0
        while True:
            if current_tree.order <= order + 1:
                return current_tree
            else:
                F = RTree.min_increase(current_tree, E)
                current_tree = F


    def condense_tree(self, L):
        """
        Adjust the tree structure to be more condense.
        :param L: a leaf node in the tree
        :return: adjusted tree
        """
        N, Q = L, []
        while N.father:
            P = N.father
            if N.num < m:
                P.delete_child(N)
                Q.append(N)
            else:
                adjust_bound(N)
            N = P
        # reinstall the entries in Q
        for q in Q:
            if isinstance(q,LeafNode):
                for elem in q.elem:
                    self.insert(elem)
            else:
                for node in q.children:
                    self.insert_node(node)
        return


class LeafNode:
    """
    Class for leaf node.
    init: leaf = LeafNode()
    """

    def __init__(self, L=[]):
        self.elem = L
        self.father = None
        self.bounding_box = self.get_bounding_box()
        self.num = len(L)
        for item in L:
            item.father = self

    def get_bounding_box(self):
        if not self.elem:
            return None
        else:
            bb_1 = self.elem[0].bounding_box
            for item in self.elem:
                bb_new = item.bounding_box
                bb_1 = bb_1.add_bb(bb_new)
            return bb_1

    def add_child(self, E):
        self.elem.append(E)
        E.father = self
        self.num += 1
        self.bounding_box = self.get_bounding_box()

    def delete_child(self, E):
        self.elem.remove(E)
        E.father = None
        seld.num -= 1
        self.bounding_box = self.get_bounding_box()

class Node:
    """
    Class for non-leaf node.
    init: node = Node()
    """

    def __init__(self, L=[]):
        self.children = L
        self.father = None
        self.bounding_box = self.get_bounding_box()
        self.num = len(L)
        self.order = self.get_order()
        for item in L:
            item.father = self

    def add_child(self, E):
        self.children.append(E)
        E.father = self
        self.bounding_box = self.get_bounding_box()
        self.num += 1

    def delete_child(self, E):
        self.children.remove(E)
        E.father = None
        self.bounding_box = self.get_bounding_box()
        self.num -= 1

    def get_bounding_box(self):
        if not self.children:
            return None
        else:
            bb_1 = self.children[0].bounding_box
            for item in self.children:
                bb_new = item.bounding_box
                bb_1 = bb_1.add_bb(bb_new)
            return bb_1

    def get_order(self):
        order, L = 0, self
        while True:
            L = L.children[0]
            order += 1
            if isinstance(L, LeafNode):
                break
        return order


if __name__ == '__main__':
    circ_1 = Circle(np.array([1, 2]), 1)
    rtree = RTree(5)
    rtree.insert(circ_1)
    print(rtree.head.elem)
    leaf_1 = LeafNode([circ_1])
