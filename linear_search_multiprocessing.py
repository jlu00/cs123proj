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
This file implements python multiprocessing library to linear_search.py.
Initially, we used a function called pool.map that automatically divides
up the data and processors behind the scene. But using this method, 
it interestingly slowed down the entire process.

So instead, after consulting with Professor Wachs, we divided up the data
into N chunks manually, and did the same with processors as well, and feeded
sub_data into each processors. This resulted in expediting linear_search by
approximately N times faster.
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

def rm_centroids_from_data(centroids, data):
'''
Inputs: list of centroids (ex: centroid_l above), 
        data = numpy array of blocks where each block is a list of 
               unique block number, lattitude, longidute, population
Outputs: same data (numpy array of blocks) but without centroids 

This function deletes the centroids
from the data so that find_nearest_block function does not return 
one of the centroids itself
'''
    for centroid in centroids:
        idx = np.where(data[:,0] == centroid[0])
        data = np.delete(data, idx, 0)
    return data

def euclidean_norm(block, centroid):
'''
Inputs: a block (list), a centroid (list)
Outputs: typical Euclidean norm

This function calculates the typical Euclidean norm. 
Initially we used explicit squaring (**)
but Nick suggested that explicit repeated multiplication is slightly faster
than using ** to sqaure.
'''
    t1 = (centroid[1] - block[1])
    t2 = (centroid[2] - block[2])
    distance = math.sqrt(t1*t1 + t2*t2)
    return [distance, block[0], block[1], block[2], block[3]]
    #reference: block[0] = num; block[1] = lat; block[2] = lon; block[3] = pop

def find_nearest_block(data, centroid, q):
'''
Inputs: data(list of blocks), a centroid, a queue
Outputs: puts the block that's nearest from the input centroid

This function calculates euclidean norm of data(numpy array of every block)
and a centroid. 
The closest block and it's information consisting of the unique ID number of the block,
lattitude, longidute, and population is put into a queue for multiprocessing
'''
    distance_list = []
    for i in range(data.shape[0]):
        distance = euclidean_norm(centroid, data[i][:])
        distance_list.append((distance, data[i][0], data[i][1], data[i][2], data[i][3]))
        #data[i][0] = num; data[i][1] = lat; data[i][2] = lon; data[i][3] = pop
    q.put(min(distance_list))

def split_data(data, chunksize):
'''
Input: data (numpy array of blocks), chunksize (integer)
Output: Numpy array of splitted data into 'chunksize' chunks

This function manually splits up the data into 'chunksize' chunks
and each will be the input to one of the processors
'''
    return np.array_split(data, chunksize)

def assign_blocks(centroids, data, processes):
'''
Inputs: list of centroids
        numpy array of blocks
        number of prcesses (integer)
Outputs: Calls graph function which plots every block

This function assigns each block to one of the centroids.
It's almost identical to using pool.map in linear_search_pool.py,
but since pool.map turned out to slow down the algorithm,
after consulting with Professor Wachs, we manually splitted up
the data into chunks and input each split chunk to each processor
which turned out to be approximately 'processes' times faster 
'''
    Districts = dc.create_districts(centroids)
    data = rm_centroids_from_data(centroids, data)

    q = Queue()

    colors_dict = get_colors(Districts)
   
    # used for stopping conditon with EPSILON
    unassigned_blocks = data.shape[0]

    while data.shape[0] != 0:
        data_splitted = split_data(data, processes)
        priority_district = dc.return_low_pop(Districts)

        for subdata in data_splitted:
            p = Process(target=find_nearest_block, args=(subdata, priority_district.centroid, q))
            p.Daemon = True
            p.start()

        for subdata in data_splitted:
            p.join()

        blocks = []
        while(not q.empty()):
            blocks.append(q.get())
        # [1:] part gets rid of the distance 
        nearest_block = list(min(blocks)[1:])

        plt.scatter(nearest_block[2], nearest_block[1], color=colors_dict[priority_district.id])

        priority_district.add_block(nearest_block, Districts) #should i get rid of distance before 
        idx = np.where(data[:,0] == nearest_block[0])
        data = np.delete(data, idx, 0)

        if (unassigned_blocks - EPSILON) == data.shape[0]:
           break

    graph(Districts, data)

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
