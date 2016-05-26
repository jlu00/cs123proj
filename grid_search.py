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
	distance = math.sqrt((centroid[1]-block[1])**2+(centroid[2]-block[2])**2)
	return distance

def debug(i_0, j_0, Grid, centroid):
	dist_list = []
	for block in Grid[int(i_0)][int(j_0)]:
		dist = euclidean_norm(centroid, block)
		dist_list.append([dist, block[0], block[1], block[2], block[3], i_0, j_0])
	heapq.heapify(dist_list)
	

#def neighborhood_to_search(centroid, tol, dim, lat, lon):
def neighborhood_to_search(centroid, tol, dim, lat, lon, Grid):
	i_0, j_0 = hash_map_index(dim, lat, lon, centroid)
	return [max(i_0-tol, 0), min(i_0+tol, dim[1]-1)], [max(j_0-tol, 0), min(j_0+tol, dim[0]-1)]

def searching_neighborhood(priority_district, tol, Grid, dim, lat, lon):
	#x_range, y_range = neighborhood_to_search(priority_district.centroid, tol, dim, lat, lon)
	x_range, y_range = neighborhood_to_search(priority_district.centroid, tol, dim, lat, lon, Grid)
	#print("x range", x_range)
	#print("y range", y_range)
	count = 0
	dist_list = []
	for i in range(x_range[0], x_range[1]+1):
 		for j in range(y_range[0], y_range[1]+1):
 			#print("ij", i, j)
 			for block in Grid[i][j]:
 				count += 1
 				dist = euclidean_norm(priority_district.centroid, block)
 				dist_list.append([dist, block[0], block[1], block[2], block[3], i, j])

	#print("counted blocks", count)
	return dist_list

def searching_all(filename, number):
	Grid, data, dim, lat, lon = build_grid(filename, number)
	
	Districts = dc.create_districts(CENTROID_L)
	unassigned_blocks = data.shape[0]
	#print(unassigned_blocks)

	#print(Districts, "districts")

	colors_dict = get_colors(Districts)
	
	while unassigned_blocks != 0:
		tol = 1
		priority_district = dc.return_low_pop(Districts)
		#print("\nASSIGNING BLOCK")
		#if not grid_is_valid(dim, lat, lon, Grid):
		#	return
		dist_list = searching_neighborhood(priority_district, tol, Grid, dim, lat, lon)
		#print(len(dist_list))

		while len(dist_list) == 0:
			tol += 1
			print("changed tolerance.")

			dist_list = searching_neighborhood(priority_district, tol, Grid, dim, lat, lon)
			#print(len(dist_list), "distlist length")
		#if not grid_is_valid(dim, lat, lon, Grid):
		#	return

		#print("length of dist list", len(dist_list))
		#print("unheaped items", dist_list[:5])
		#print('\n')
		heapq.heapify(dist_list)
		#print("heaped items", dist_list[:5])
		#print('\n')
		#print(centroid_l)
		#print(data.shape[0])

		add_block = dist_list[0]
		priority_district.add_block(add_block[1:-2], Districts)
		#print("i j:", add_block[5], add_block[6])

		Grid[int(add_block[5])][int(add_block[6])].remove(add_block[1:-2])
		#print("blocks", add_block)
		plt.scatter(add_block[3], add_block[2], color=colors_dict[priority_district.id])
		unassigned_blocks -= 1
		#if not grid_is_valid(dim, lat, lon, Grid):
		#	return

		#if unassigned_blocks == (data.shape[0] - EPSILON):
		#	break
		print(unassigned_blocks)
		#print("population of priority district", priority_district.population)
		#print("which district", priority_district.id)
	graph(Districts, data)

def get_colors(Districts):
	colors_dict = {}
	colors = itertools.cycle(["b", "g", "r", "c", "m", "y"])
	for district in Districts:
		c = next(colors)
		colors_dict[district.id] = c
	return colors_dict

def graph(Districts, data):
	#plt.scatter(data[:, 2], data[:, 1], color='k')

	'''
	colors = itertools.cycle(["b", "g", "r", "c", "m", "y"])
	
	for district in Districts:
		#print("change district")
		#print("blocks in district", len(district.blocks))
		#print("population in district", district.population)
		c = next(colors)
		for block in district.blocks:
			plt.scatter(block[2], block[1], color=c)
	'''
	xx = []
	yy = []
	for c in CENTROID_L:
		xx.append(c[2])
		yy.append(c[1])

	plt.scatter(xx, yy, color='w')#, marker='o')
	plt.savefig(str(EPSILON)+".png")
	plt.show()

#data = np.genfromtxt("IL.csv", delimiter=',', skip_header=True)
#plt.scatter(data[:, 2], data[:, 1])
#plt.savefig("raw.png")

if __name__ == "__main__":
	if int(sys.argv[2]) <= 1:
		print("Not enough number of districts.")
		sys.exit(1)

	CENTROID_L = find_random_centroids(sys.argv[1], int(sys.argv[2]))
	DISTRICTS = dc.create_districts(CENTROID_L)
	EPSILON = int(sys.argv[3])
	searching_all(sys.argv[1], int(sys.argv[2]))
