import csv
import random

def find_random_centroids(filename, number):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        random_blocks = random.sample(list(reader), number)
        centroid_list = list(random_blocks)
        centroids = []
        for c in centroid_list:
            formatted_c = []
            for d in c:
                formatted_c.append(float(d))
            centroids.append(formatted_c)

    return centroids
