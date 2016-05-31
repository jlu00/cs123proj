import csv
import random
import sys
from grid import hash_map_index, create_grid

def find_random_centroids(filename, number):
    random.seed(0)
    hash_list = []
    centroid_list = []
    dim, lat, lon, data = create_grid(filename, number)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        reader = list(reader)
        for i in range(number):
            random_block = random.sample(reader, 1)
            hm_tuple = hash_map_index(dim, lat, lon, random_block)
            if hm_tuple not in hash_list:
                hash_list.append(hm_tuple)
                print(hm_tuple)
        centroids = []
        for c in centroid_list:
            formatted_c = []
            for d in c:
                formatted_c.append(float(d))
            centroids.append(formatted_c)

    return centroids

'''
def find_random_centroids(filename, number):
    random.seed(0)
    hash_dict = {}
    centroid_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        reader = list(reader)
        for i in range(number):
            random_block = random.sample()
        random_blocks = random.sample(reader, int(number))


        centroid_list = list(random_blocks)
        centroids = []
        for c in centroid_list:
            formatted_c = []
            for d in c:
                formatted_c.append(float(d))
            centroids.append(formatted_c)

    return centroids
'''