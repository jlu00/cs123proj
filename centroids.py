import csv
import random
import sys

def find_random_centroids(filename, number):
    random.seed(0)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        random_blocks = random.sample(list(reader), int(number))
        centroid_list = list(random_blocks)
        centroids = []
        for c in centroid_list:
            formatted_c = []
            for d in c:
                formatted_c.append(float(d))
            centroids.append(formatted_c)

    return centroids
