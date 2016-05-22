import csv
import random

def find_random_centroids(filename, number):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        random_blocks = random.sample(list(reader), number)
        print(random_blocks)
        centroid_list = list(random_blocks)
        centroids = []
        for c in centroid_list:
            formatted_c = []
            for d in c:
                formatted_c.append(float(d))
            centroids.append(formatted_c)

        #centroid_list = [float(i) for i in centroid_list]
    return centroids

def format_centroids(centroid_list):
	for item in centroid_list:
		print(item[2][1:] + "," + item[1])