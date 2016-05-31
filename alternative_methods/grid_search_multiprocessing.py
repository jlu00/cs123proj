import district_class as dc
import numpy as np
import math
from grid import build_grid, hash_map_index, grid_is_valid, find_random_centroids
from centroids import find_random_centroids
import heapq 
import matplotlib.pyplot as plt
import itertools
import sys
from functools import partial
from multiprocessing import Process, Queue
'''
This file is almost identical to grid_search,py
but it implements multiprocessing method that
we tested in linear_search_multiprocessing.py.

However, interestingly, this method turned out to be
slower than regular gird_search.py even though
multiprocessing method did expedite linear_search.py significantly.

We speculate that the reason is since grid_search.py
is already significnatly optimized and only loops through
a small subset of the entire data (numpy array of blocks), 
there might not be enough blocks to merit from launching
several processors and dividing up the data into chunks.
(i.e. dividing up the data and launching several processors
might be more expensive)
'''

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

def find_nearest_block(data, centroid, q):
	distance_list = []
	for i in range(data.shape[0]):
		distance = euclidean_norm(centroid, data[i][:])
		distance_list.append((distance, data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5]))

	q.put(min(distance_list))

def neighborhood_to_search(centroid, tol, dim, lat, lon, Grid):
	i_0, j_0 = hash_map_index(dim, lat, lon, centroid)
	return [max(i_0-tol, 0), min(i_0+tol, dim[1]-1)], [max(j_0-tol, 0), min(j_0+tol, dim[0]-1)]


def searching_neighborhood(priority_district, tol, Grid, dim, lat, lon):
'''
Inputs: priority_district, tolerance, Grid, dimension, lat, lon
Outputs: numpry array of the subset of the data (list of blocks)

This function returns numpy array of blocks that is a much smaller subset of
the original dataset.
'''
	x_range, y_range = neighborhood_to_search(priority_district.centroid, tol, dim, lat, lon, Grid)
	count = 0

	subset = []
	for i in range(x_range[0], x_range[1]+1):
		for j in range(y_range[0], y_range[1]+1):
			for block in Grid[i][j]:
				subset.append(block)

	return np.asarray(subset)

'''
This function implements the multiprocessing technique that was 
tested in linear_search_multiprocessing.py. 
Even though the technique expedited linear_search.py by approximately
the number of processors times faster, it turned out that 
implementing the same technique to grid_search.py slows down the code.
'''
def searching_all(filename, number):
	Grid, data, dim, lat, lon = build_grid(filename, number)
	
	Districts = dc.create_districts(CENTROID_L)
	unassigned_blocks = data.shape[0]

	q = Queue()
	processes = 5
	colors_dict = get_colors(Districts)
	
	while unassigned_blocks != 0:
		tol = 1
		priority_district = dc.return_low_pop(Districts)

		subset = searching_neighborhood(priority_district, tol, Grid, dim, lat, lon)
		print(subset.shape)

		split_subset = np.array_split(subset, processes)

		for subdata in split_subset:
			p = Process(target=find_nearest_block, args=(subdata, priority_district.centroid, q))
			p.Daemon = True
			p.start()

		for subdata in split_subset:
			p.join()

		while q.empty():
			tol += 1
			print("changed tolerance.")
			subset = searching_neighborhood(priority_district, tol, Grid, dim, lat, lon)

		blocks = []
		while(not q.empty()):
			blocks.append(q.get())
		nearest_block = list(min(blocks)[1:])

		priority_district.add_block(nearest_block[:-2], Districts)

		Grid[int(nearest_block[-2])][int(nearest_block[-1])].remove(nearest_block[:-2])
		plt.scatter(nearest_block[2], nearest_block[1], color=colors_dict[priority_district.id])
		unassigned_blocks -= 1

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
	for c in CENTROID_L:
		xx.append(c[2])
		yy.append(c[1])

	plt.scatter(xx, yy, color='w')
	plt.savefig(str(EPSILON)+".png")
	plt.show()


if __name__ == "__main__":
	if int(sys.argv[2]) <= 1:
		print("Not enough number of districts.")
		sys.exit(1)

	CENTROID_L = find_random_centroids(sys.argv[1], int(sys.argv[2]))
	DISTRICTS = dc.create_districts(CENTROID_L)
	EPSILON = int(sys.argv[3])
	searching_all(sys.argv[1], int(sys.argv[2]))
