import numpy as np
from Shapes import Rectangle, Circle, Line, Point, BoundingBox, ShapeUnion
from RTree import RTree, Node, LeafNode

node_lst = []
for i in range(500):
    circ = Circle(np.array([i, i]), 1, id={'name':'node'+ str(i)})
    node_lst.append(circ)

#for item in node_lst:
#   print(item.id['name']+':')
#   item.bounding_box.print()


import random
random.seed(42)
r_tree = RTree(M=3)
i = 1
for item in node_lst:
    r_tree.insert(item)
head = r_tree.head

print('\n')
head.bounding_box.print()
for c in head.children:
    c.bounding_box.print()
    print('\nlayer 1: ', c)
    if isinstance(c, LeafNode):
        for k in c.elem:
            print(k.id['name'])
    else:
        for e in c.children:
            print('   layer 2: ', e)
            e.bounding_box.print()
            if isinstance(e, Node):
                print('Node: ', e.children)
            else:
                for elem in e.elem:
                    print(elem.id['name'])
print('\n')
print(RTree.find_leaf(head, node_lst[1]))
print(RTree.find_leaf(head, Circle(np.array([1000, 1000]), 1, id={'name':'node'+ str(i)})))
