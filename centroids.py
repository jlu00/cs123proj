import csv
import random

def find_random_centroids(filename, number):
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		random_blocks = random.sample(list(reader), number)
		centroid_list = list(random_blocks)
	return centroid_list

def format_centroids(centroid_list):
	for item in centroid_list:
		print(item[2][1:] + "," + item[1])