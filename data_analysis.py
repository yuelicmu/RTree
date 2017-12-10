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
