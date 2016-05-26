from mrjob.job import MRjob
import numpy as np
import heapq
import random
import csv
import math
import matplotlib.pyplot as plt
import os


class MRStates(MRJob):
	def mapper(self, _, line):
		stateline = line.split(", ")
		redistrict("statecsv." + stateline[0], state[1])


def redistrict(filename, number):
	CENTROID_L = find_random_centroids(filename, number)
	DISTRICTS = dc.create_districts(CENTROID_L)
	STATENAME = filename[0:2]
	searching_all(filename, number)

def euclidean_norm(centroid, block):
	distance = math.sqrt((centroid[1]-block[1])**2+(centroid[2]-block[2])**2)
	return distance
def neighborhood_to_search(centroid, tol, dim, lat, lon, Grid):
	i_0, j_0 = hash_map_index(dim, lat, lon, centroid)
	return [max(i_0-tol, 0), min(i_0+tol, dim[1]-1)], [max(j_0-tol, 0), min(j_0+tol, dim[0]-1)]

def searching_neighborhood(priority_district, tol, Grid, dim, lat, lon):
	x_range, y_range = neighborhood_to_search(priority_district.centroid, tol, dim, lat, lon, Grid)
	count = 0
	dist_list = []
	for i in range(x_range[0], x_range[1]+1):
 		for j in range(y_range[0], y_range[1]+1):
 			for block in Grid[i][j]:
 				count += 1
 				dist = euclidean_norm(priority_district.centroid, block)
 				dist_list.append([dist, block[0], block[1], block[2], block[3], i, j])
	return dist_list

def searching_all(filename, number):
	Grid, data, dim, lat, lon = build_grid(filename, number)
	
	Districts = dc.create_districts(CENTROID_L)
	unassigned_blocks = data.shape[0]

	colors_dict = get_colors(Districts)
	
	while unassigned_blocks != 0:
		tol = 1
		priority_district = dc.return_low_pop(Districts)
		dist_list = searching_neighborhood(priority_district, tol, Grid, dim, lat, lon)

		while len(dist_list) == 0:
			tol += 1
			print("changed tolerance.")

			dist_list = searching_neighborhood(priority_district, tol, Grid, dim, lat, lon)
		#heapq.heapify(dist_list)

		add_block = min(dist_list)
		priority_district.add_block(add_block[1:-2], Districts)
		Grid[int(add_block[5])][int(add_block[6])].remove(add_block[1:-2])
		plt.scatter(add_block[3], add_block[2], color=colors_dict[priority_district.id])
		unassigned_blocks -= 1
		print(unassigned_blocks)
	graph(Districts, data)

def get_colors(Districts):
	colors_dict = {}
	colors = itertools.cycle(["b", "g", "r", "c", "m", "y"])
	for district in Districts:
		c = next(colors)
		colors_dict[district.id] = c
	return colors_dict

def graph(districts, data):
	#plt.scatter(data[:, 2], data[:, 1], color='k')
	xx = []
	yy = []
	for c in CENTROID_L:
		xx.append(c[2])
		yy.append(c[1])

	plt.scatter(xx, yy, color='w')
	plt.savefig(STATENAME+".png")
	plt.show()

def create_grid(filename, number):
	'''
	[ (id, lat, long, pop),
	   (id, lat, long, pop)]

	[ [id, lat, long, pop],
	  [id, lat, long, pop] ]
	'''
	data = np.genfromtxt(filename, delimiter=',', skip_header=True)
	CB_Per_GB = (data.shape[0]/number)*(2/9)
	
	eps = 0.00000001
	max_id, max_lat, max_lon, pop = data.max(axis=0)
	min_id, min_lat, min_lon, min_pop = data.min(axis=0)
	max_lon += eps
	max_lat += eps
	min_lat -= eps
	min_lon -= eps

	blocks = data.shape[0]/CB_Per_GB
	lon_to_lat =  (max_lon - min_lon) / (max_lat - min_lat)
	y_num = math.sqrt(blocks/lon_to_lat)
	x_num = blocks/y_num

	return [int(math.ceil(x_num)), int(math.ceil(y_num))], [min_lat, max_lat], [min_lon, max_lon], data

def hash_map_index(dim, lat, lon, block):
	x_size = (lon[1] - lon[0]) / dim[0]
	y_size = (lat[1] - lat[0]) / dim[1]
	_j = int((float(block[2]) - lon[0]) / x_size) 
	_i = int((float(block[1]) - lat[0]) / y_size) 
	j = (dim[0]-1) - _j
	i = (dim[1]-1) - _i
	return i, j

def find_random_centroids(filename, number):
    random.seed(0)
    hash_list = []
    centroid_list = []
    dim, lat, lon, data = create_grid(filename, number)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        reader = list(reader)
        start = 0
        while start < number:
            random_block = random.sample(reader, 1)[0]
            hm_tuple = hash_map_index(dim, lat, lon, random_block)
            if hm_tuple not in hash_list:
                hash_list.append(hm_tuple)
                centroid_list.append(random_block)
                start += 1
        centroids = []
        for c in centroid_list:
            formatted_c = []
            for d in c:
                formatted_c.append(float(d))
            centroids.append(formatted_c)
    return centroids

	
def grid_is_valid(dim, lat, lon, Grid):
	count = 0
	for x in range(dim[1]):
		for y in range(dim[0]):
			for block in Grid[x][y]:
				count += 1
				i, j = hash_map_index(dim, lat, lon, block)
				if x != i:
					return False
				if y != j:
					return False
	return True

def build_grid(filename, number):
	dim, lat, lon, data = create_grid(filename, number)

	Master_Grid = []
	for r in range(dim[1]):
		row = []
		for c in range(dim[0]):
			row.append([])
		Master_Grid.append(row)
	count = 0
	for item in data:
		count += 1
		i, j = hash_map_index(dim, lat, lon, item)
		Master_Grid[i][j].append(item.tolist())
	if not grid_is_valid(dim, lat, lon, Master_Grid):
		return
	return Master_Grid, data, dim, lat, lon

if __name__ == '__main__':
	MRStates.run()