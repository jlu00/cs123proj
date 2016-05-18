import district_class as dc
import numpy as np
import math
import heapq
from multiprocessing import pool

data = np.genfromtxt('IL.csv', delimiter=',', skip_header=True)

centroid_l = [[1,-089.192249,41.326240,39],
[10,-089.143829,38.530543,18],
[30,-087.693154,41.897982,374],
[40,-088.250446,41.992887,51],
[50,-087.713445,41.510284,6],
[60,-089.048809,40.119928,1],
[70,-087.679184,39.628681,23],
[80,-088.423013,41.367358,90],
[90,-088.792693,38.259951,17],
[100,-088.332532,41.543430,5],
[110,-089.110784,41.542103,26],
[120,-088.443526,42.335221,54],
[130,-090.055654,38.823547,28],
[140,-090.301684,38.296993,4],
[150,-088.736112,40.014611,0],
[160,-089.921620,38.578698,38],
[170,-089.792060,42.490470,39],
[180,-088.086253,38.140670,4],
[190,-089.022686,37.722950,0]]

Districts = dc.create_districts(centroid_l, 1)

def euclidean_norm(centroid, block):
    distance = math.sqrt((centroid[1]-block[1])**2+(centroid[2]-block[2])**2)
    return distance


dist_list = []
while data.shape[0] != 0:
    priority_district = dc.return_low_pop(Districts)

    for i in range(data.shape[0]):
        dist = euclidean_norm(priority_district.centroid, data[i][:])
        dist_list.append((dist, data[i][0], data[i][1], data[i][2], data[i][3]))
        #data[i][0] = num
        #data[i][1] = lat
        #data[i][2] = lon
        #data[i][0] = pop

    heapq.heapify(dist_list)
    priority_district.add_block(dist_list[0])
    idx = np.where(data[:,0] == dist_list[0][1])
    data = np.delete(data, idx[0][0], 0)