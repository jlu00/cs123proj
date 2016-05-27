import district_class as dc
import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
import itertools
import sys
from functools import partial
from multiprocessing import Pool, Process, Queue

EPSILON = 500

data = np.genfromtxt('IL.csv', delimiter=',', skip_header=True)
#data = np.genfromtxt('ILsubset.csv', delimiter=',', skip_header=True)

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
centroid_l = [[1,+39.825048,-090.952683,27],
[275,+40.045300,-091.083620,11],
[499,+40.013597,-090.960674,16]]
'''
# after creating districts, I should delete the centroids from the data
# otherwise the least distance block will be centroid itself
def rm_centroids_from_data(centroids, data):
    for centroid in centroids:
        idx = np.where(data[:,0] == centroid[0])
        data = np.delete(data, idx, 0)
    return data

def euclidean_norm(block, centroid):
    t1 = (centroid[1] - block[1])
    t2 = (centroid[2] - block[2])
    distance = math.sqrt(t1*t1 + t2*t2)
    return [distance, block[0], block[1], block[2], block[3]]
    #block[0] = num; block[1] = lat; block[2] = lon; block[3] = pop

def find_nearest_block(data, centroid, q):
    distance_list = []
    for i in range(data.shape[0]):
        distance = euclidean_norm(centroid, data[i][:])
        distance_list.append((distance, data[i][0], data[i][1], data[i][2], data[i][3]))
        #data[i][0] = num; data[i][1] = lat; data[i][2] = lon; data[i][3] = pop
    #return min(distance_list)
    q.put(min(distance_list))

def split_data(data, chunksize):
    return np.array_split(data, chunksize)

def assign_blocks(centroids, data):
    Districts = dc.create_districts(centroids)
    data = rm_centroids_from_data(centroids, data)

    q = Queue()

    #pool = Pool(processes = None)
    processes = 5


    colors_dict = get_colors(Districts)
    
    # stopping conditon with EPSILON
    unassigned_blocks = data.shape[0]

    while data.shape[0] != 0:

        data_splitted = split_data(data, processes)

        priority_district = dc.return_low_pop(Districts)

        #norm_one_arg = partial(euclidean_norm, centroid=priority_district.centroid)
        #distance_list = pool.imap(norm_one_arg, data, chunksize=10000)
        for subdata in data_splitted:
            p = Process(target=find_nearest_block, args=(subdata, priority_district.centroid, q))
            p.Daemon = True
            p.start()

        for subdata in data_splitted:
            p.join()

        #nearest_block = min(distance_list) #[1:]
        blocks = []
        while(not q.empty()):
            blocks.append(q.get())
        nearest_block = list(min(blocks)[1:])

        #print("nearest block: ", nearest_block)

        plt.scatter(nearest_block[2], nearest_block[1], color=colors_dict[priority_district.id])

        priority_district.add_block(nearest_block, Districts) #should i get rid of distance before 
        idx = np.where(data[:,0] == nearest_block[0])
        #print("idx: ", idx)
        data = np.delete(data, idx, 0)

        #print("length of data: ",data.shape[0])


        if (unassigned_blocks - EPSILON) == data.shape[0]:
           break

    graph(Districts, data)

    #pool.close()
    #pool.join()

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
