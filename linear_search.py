import district_class as dc
import numpy as np
import math
import heapq
from multiprocessing import pool

'''
This file implements the most basic method of our algorithm.


'''
data = np.genfromtxt('IL.csv', delimiter=',', skip_header=True)

centroid_l = [[5,+39.810985,-090.925895,6],
[60505,+41.781883,-087.684522,111],
[120990,+41.952518,-088.196756,43],
[175250,+41.884529,-088.310647,30],
[213505,+40.159843,-089.342779,81],
[47455,+38.837947,-090.056454,0],
[258010,+41.094140,-090.936430,15],
[268540,+42.136809,-089.287520,25],
[100310,+39.265350,-088.026622,15],
[270390,+42.008476,-089.297788,2],
[270450,+41.971581,-089.443183,0],
[286895,+38.591100,-088.139386,6],
[298995,+38.621083,-089.901350,32],
[317725,+42.431654,-089.750755,8],
[324765,+40.144836,-087.594293,102],
[336465,+38.104917,-088.133699,8],
[344276,+41.624047,-087.923115,170],
[351295,+41.284644,-088.114612,114],
[365115,+40.629956,-089.275667,12]]

Districts = dc.create_districts(centroid_l, 1)

'''
Inputs: list of centroids (ex: centroid_l above), 
        data = numpy array of blocks where each block is a list of 
               unique block number, lattitude, longidute, population
Outputs: same data (numpy array of blocks) but without centroids 

This function deletes the centroids
from the data so that find_nearest_block function does not return 
one of the centroids itself
'''
def rm_centroids_from_data(centroids, data):
    for centroid in centroids:
        idx = np.where(data[:,0] == centroid[0])
        data = np.delete(data, idx[0][0], 0)
    return data

'''
Inputs: a block (list), a centroid (list)
Outputs: typical Euclidean norm

This function calculates the typical Euclidean norm. 
Initially we used explicit squaring (**)
but Nick suggested that explicit repeated multiplication is slightly faster
than using ** to sqaure.
'''
def euclidean_norm(centroid, block):
    t1 = (centroid[1] - block[1])
    t2 = (centroid[2] - block[2])
    distance = math.sqrt(t1*t1 + t2*t2)
    return distance

'''
Inputs: list of centroids
        numpy array of blocks
        number of prcesses (integer)
Outputs: Calls graph function which plots every block

This function assigns each block to one of the centroids.
1) remove centroids from the input data
2) identifies the district with smallest population
3) calculates the distance between the centroid of smallest population
   and every block in data (which is numpy array of blocks)
4) identify the nearest block from the centroid 
5) the district absorbs the nearest block, recalculates the population
6) repeat 2~5 until every block has been assigned
'''
def assign_blocks(centroids, data):
    data = rm_centroids_from_data(centroid_l, data)

    while data.shape[0] != 0:
        dist_list = []
        priority_district = dc.return_low_pop(Districts)
        for i in range(data.shape[0]):
            dist = euclidean_norm(priority_district.centroid, data[i][:])
            dist_list.append((dist, data[i][0], data[i][1], data[i][2], data[i][3]))
            #data[i][0] = num
            #data[i][1] = lon
            #data[i][2] = lat
            #data[i][0] = pop
        nearest_block = min(dist_list)
        priority_district.add_block(nearest_block)
        idx = np.where(data[:,0] == nearest_block[1])
        data = np.delete(data, idx[0][0], 0)
