import district_class as dc
import numpy as np
import math
import heapq
from functools import partial
from multiprocessing import Pool

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
    return distance
    #return [distance, block[0], block[1], block[2], block[3]]
    #block[0] = num; block[1] = lat; block[2] = lon; block[3] = pop

def find_nearest_block(data, centroid):    
    distance_list = []
    for i in range(data.shape[0]):
        distance = euclidean_norm(centroid, data[i][:])
        distance_list.append((distance, data[i][0], data[i][1], data[i][2], data[i][3]))
        #data[i][0] = num; data[i][1] = lat; data[i][2] = lon; data[i][3] = pop

    return min(distance_list)

def assign_blocks(centroids, data):
    Districts = dc.create_districts(centroids, 1)
    data = rm_centroids_from_data(centroids, data)

    #pool = Pool(processes = 8)

    while data.shape[0] != 0:
        print("number of block is ", data.shape[0])
        priority_district = dc.return_low_pop(Districts)

        nearest_block = find_nearest_block(data, priority_district.centroid)[1:]
        #find_nearest_block_one_arg = partial(find_nearest_block, centroid=priority_district.centroid)
        #nearest_block = pool.map(find_nearest_block_one_arg, data)       

        #distance_list = pool.map(partial(euclidean_norm, centroid=priority_district.centroid, data)
        #heapq.heapify(distance_list)
        #nearest_block = distance_list[0][1:]
        print("nearest block is ", nearest_block)

        priority_district.add_block(nearest_block) #should i get rid of distance before 
        idx = np.where(data[:,0] == nearest_block[0])
        print("index of block that's deleted is :",idx)
        data = np.delete(data, idx, 0)

    #pool.close()
    #pool.join()

assign_blocks(centroid_l, data)


