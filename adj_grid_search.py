import district_class as dc
import numpy as np
import math
from grid import build_grid, hash_map_index, grid_is_valid, find_random_centroids
#from centroids import find_random_centroids
import heapq 
import matplotlib.pyplot as plt
import itertools
import sys


def euclidean_norm(centroid, block):
	t1 = (centroid[1] - block[1])
	t2 = (centroid[2] - block[2])
	distance = math.sqrt(t1*t1 + t2*t2)
	return distance

def debug(i_0, j_0, Grid, centroid):
	dist_list = []
	for block in Grid[int(i_0)][int(j_0)]:
		dist = euclidean_norm(centroid, block)
		dist_list.append([dist, block[0], block[1], block[2], block[3], i_0, j_0])
	heapq.heapify(dist_list)
	
def neighborhood_to_search(priority_district, dim, lat, lon):
#def neighborhood_to_search(centroid, tol, dim, lat, lon, Grid):
	centroid = priority_district.centroid
	tol = priority_district.tolerance
	i_0, j_0 = hash_map_index(dim, lat, lon, centroid)
	return [max(i_0-tol, 0), min(i_0+tol, dim[1]-1)], [max(j_0-tol, 0), min(j_0+tol, dim[0]-1)]

def search_cell(Grid, i, j, dist_list, priority_district):
	for block in Grid[i][j]:
		dist = euclidean_norm(priority_district.centroid, block)
		dist_list.append([dist, block[0], block[1], block[2], block[3], i, j])

def searching_neighborhood(priority_district, Grid, dim, lat, lon):
	x_range, y_range = neighborhood_to_search(priority_district, dim, lat, lon)
	
	dist_list = []
	for i in range(x_range[0], x_range[1]+1):
		for j in range(y_range[0], y_range[1]+1):
			search_cell(Grid, i, j, dist_list, priority_district)

	return dist_list

def searching_all(filename, number):
	Grid, data, dim, lat, lon = build_grid(filename, number)
	Districts = dc.create_districts(CENTROID_L, Grid, dim, lat, lon)
	colors_dict = get_colors(Districts)

	unassigned_blocks = data.shape[0]

	while unassigned_blocks != 0:
		priority_district = dc.return_low_pop(Districts)

		while priority_district.neighborhood.shape[0] == 0:
			priority_district.increment_tolerance(Grid, dim, lat, lon)

		add_block = priority_district.neighborhood[0].tolist()
		priority_district.remove_neighborhood_block(add_block)

		if add_block[1:-2] not in Grid[int(add_block[5])][int(add_block[6])]:
			heapq.heappush(Districts, priority_district)
			continue
			
		priority_district.add_block(add_block[1:-2], Districts)

		Grid[int(add_block[5])][int(add_block[6])].remove(add_block[1:-2])

		plt.scatter(add_block[3], add_block[2], color=colors_dict[priority_district.id])
		unassigned_blocks -= 1

	graph(Districts)

def get_colors(Districts):
	colors_dict = {}
	colors = itertools.cycle(["b", "g", "r", "c", "m", "y"])
	for district in Districts:
		c = next(colors)
		colors_dict[district.id] = c
	return colors_dict

def graph(Districts):
	#plt.scatter(data[:, 2], data[:, 1], color='k')

	xx = []
	yy = []
	for c in CENTROID_L:
		xx.append(c[2])
		yy.append(c[1])

	plt.scatter(xx, yy, color='w')#, marker='o')
	plt.savefig("adjusted_illinois.png")
	plt.show()

if __name__ == "__main__":
	if int(sys.argv[2]) <= 1:
		print("Not enough number of districts.")
		sys.exit(1)

	CENTROID_L = find_random_centroids(sys.argv[1], int(sys.argv[2]))
	#DISTRICTS = dc.create_districts(CENTROID_L)
	#EPSILON = int(sys.argv[3])
	searching_all(sys.argv[1], int(sys.argv[2]))
