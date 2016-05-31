import district_class as dc
import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
import itertools
import sys
from functools import partial
from multiprocessing import Pool, Process, Queue

'''
This file is almost identical to linear_search_multiprocessing
but uses pool.map method instead of manually splitted up the data.
It turned out that this method slows down the algorithm rathr than expediting.
'''

EPSILON = 500
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
        data = np.delete(data, idx, 0)
    return data

'''
Inputs: a block (list), a centroid (list)
Outputs: typical Euclidean norm with the input block's information

This function calculates the typical Euclidean norm. 
Initially we used explicit squaring (**)
but Nick suggested that explicit repeated multiplication is slightly faster
than using ** to sqaure.
'''
def euclidean_norm(block, centroid):
    t1 = (centroid[1] - block[1])
    t2 = (centroid[2] - block[2])
    distance = math.sqrt(t1*t1 + t2*t2)
    return [distance, block[0], block[1], block[2], block[3]]
    #reference: block[0] = num; block[1] = lat; block[2] = lon; block[3] = pop

'''
Inputs: data(list of blocks), a centroid, a queue
Outputs: puts the block that's nearest from the input centroid

This function calculates euclidean norm of data(numpy array of every block)
and a centroid. 
The closest block and it's information consisting of the unique ID number of the block,
lattitude, longidute, and population is put into a queue for multiprocessing
'''
def find_nearest_block(data, centroid, q):
    distance_list = []
    for i in range(data.shape[0]):
        distance = euclidean_norm(centroid, data[i][:])
        distance_list.append((distance, data[i][0], data[i][1], data[i][2], data[i][3]))
        #data[i][0] = num; data[i][1] = lat; data[i][2] = lon; data[i][3] = pop
    q.put(min(distance_list))

'''
Inputs: data (numpy array of blocks), chunksize (integer)
Outputs: Numpy array of splitted data into 'chunksize' chunks

This function manually splits up the data into 'chunksize' chunks
and each will be the input to one of the processors
'''
def split_data(data, chunksize):
    return np.array_split(data, chunksize)

'''
Inputs: list of centroids
        numpy array of blocks
Outputs: Calls graph function which plots every block

This function assigns each block to one of the centroids.


This commented secion is almost identical to 
the same function in linear_search_multiprocessing.py
but uses pool.map function as described in the beginning of this file.
For some reason, this method slowed down rather than expediting.
'''
def assign_blocks(centroids, data):
    Districts = dc.create_districts(centroids)
    data = rm_centroids_from_data(centroids, data)

    # this will generate number of pool workers based on the 
    # number of cores
    pool = Pool(processes = None)

    colors_dict = get_colors(Districts)
    
    # stopping conditon with EPSILON
    unassigned_blocks = data.shape[0]

    while data.shape[0] != 0:
        priority_district = dc.return_low_pop(Districts)

        # pool.map only accepts function with one argument,
        # so we use Partial from Functools library to recreate
        # euclidiean_norm function with one argument (data)
        norm_one_arg = partial(euclidean_norm, centroid=priority_district.centroid)
        distance_list = pool.map(norm_one_arg, data, chunksize=10000)

        nearest_block = min(distance_list)[1:]

        plt.scatter(nearest_block[2], nearest_block[1], color=colors_dict[priority_district.id])

        priority_district.add_block(nearest_block, Districts) #should i get rid of distance before 
        idx = np.where(data[:,0] == nearest_block[0])

        data = np.delete(data, idx, 0)

    graph(Districts, data)

    pool.close()
    pool.join()


def get_colors(Districts):
    colors_dict = {}
    colors = itertools.cycle(["b", "g", "r", "c", "m", "y"])
    for district in Districts:
        c = next(colors)
        colors_dict[district.id] = c
    return colors_dict

def graph(Districts, data):
    xx = []
    yy = []
    for c in centroid_l:
        xx.append(c[2])
        yy.append(c[1])

    plt.scatter(xx, yy, color='w')#, marker='o')
    plt.savefig(str(EPSILON)+".png")
    plt.show()


assign_blocks(centroid_l, data)
