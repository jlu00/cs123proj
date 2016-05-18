import district_class as dc
import numpy as np
import math
from grid import build_grid, hash_map_index
import heapq 
import matplotlib.pyplot as plt
import itertools

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

Districts = dc.create_districts(centroid_l, 1)
Grid, data, dim, lat, lon = build_grid("IL.csv")

def euclidean_norm(centroid, block):
	distance = math.sqrt((centroid[1]-block[1])**2+(centroid[2]-block[2])**2)
	return distance

def neighborhood_to_search(centroid, tol):
	i_0, j_0 = hash_map_index(dim, lat, lon, centroid)
	return [max(i_0-tol, 0), min(i_0+tol, len(Grid[0]))], [max(j_0-tol, 0), min(j_0+tol, len(Grid))]

def searching_neighborhood(priority_district, tol):
	y_range, x_range = neighborhood_to_search(priority_district.centroid, tol)
 	dist_list = []
 	for i in x_range:
 		for j in y_range:
 			for block in Grid[i][j]:
				dist = euclidean_norm(priority_district.centroid, block)
				dist_list.append([dist, block[0], block[1], block[2], block[3], i, j])
	return dist_list

def searching_all(filename):
	unassigned_blocks = data.shape[0]
	print(unassigned_blocks)
	while unassigned_blocks != 0:
		tol = 1
	 	priority_district = dc.return_low_pop(Districts)
	 	dist_list = searching_neighborhood(priority_district, tol)
		
		while dist_list == 0:
			tol += 1
			dist_list = searching_neighborhood(priority_district, tol)

		heapq.heapify(dist_list)

		add_block = dist_list[0]
		priority_district.add_block(add_block[1:-2])
		
		Grid[int(add_block[5])][int(add_block[6])].remove(add_block[1:-2])
		unassigned_blocks -= 1

		if unassigned_blocks == (366138 - 100):
			break
		print(unassigned_blocks)

def graph(Districts):
	colors = itertools.cycle(["b", "g", "r", "c", "m", "y", "k"])
	for district in Districts:
		print("change district")
		print("blocks in district", len(district.blocks))
		print("population in district", district.population)
		c = next(colors)
		for block in district.blocks:
			plt.scatter(block[1], block[2], color=c)

	plt.show()

searching_all("IL.csv")
graph(Districts)