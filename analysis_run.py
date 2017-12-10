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
