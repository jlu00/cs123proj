import district_class as dc
import numpy as np
import math
from grid import build_grid, hash_map_index
from centroids import find_random_centroids
import heapq 
import matplotlib.pyplot as plt
#import matplotlib as plt
import itertools

'''
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
'''

filename= "NV.csv"
number = 1
global_epsilon = 1000

centroid_l = find_random_centroids(filename, number)

def euclidean_norm(centroid, block):
	distance = math.sqrt((centroid[1]-block[1])**2+(centroid[2]-block[2])**2)
	return distance

def neighborhood_to_search(centroid, tol, dim, lat, lon):
	i_0, j_0 = hash_map_index(dim, lat, lon, centroid)
	#x_size = (lon[1] - lon[0]) / dim[0]
	#y_size = (lat[1] - lat[0]) / dim[1]

	#print("\n actual location", centroid[1], centroid[2])
	#print("dim", dim)
	#print("x size", x_size)
	#print("y size", y_size)
	#print("lat", lat)
	#print("lon", lon)
	#print("centroid i_0, j_0", i_0, j_0)
	#print("j_0", j_0)
	return [max(i_0-tol, 0), min(i_0+tol, dim[1]-1)], [max(j_0-tol, 0), min(j_0+tol, dim[0]-1)]


def searching_neighborhood(priority_district, tol, Grid, dim, lat, lon):
	x_range, y_range = neighborhood_to_search(priority_district.centroid, tol, dim, lat, lon)
	#print("x range", x_range)
	#print("y range", y_range)
	dist_list = []
	for i in x_range:
 		for j in y_range:
 			for block in Grid[i][j]:
 				#print("ij", i, j)
 				dist = euclidean_norm(priority_district.centroid, block)
 				dist_list.append([dist, block[0], block[1], block[2], block[3], i, j])
	return dist_list

def searching_all(filename):
	Grid, data, dim, lat, lon = build_grid(filename, number)
	print(len(Grid[0]), dim[0])
	print(len(Grid), dim[1])
	Districts = dc.create_districts(centroid_l, 1)
	unassigned_blocks = data.shape[0]
	#print(unassigned_blocks)

	while unassigned_blocks != 0:
		tol = 1
		priority_district = dc.return_low_pop(Districts)
		dist_list = searching_neighborhood(priority_district, tol, Grid, dim, lat, lon)
		
		while len(dist_list) == 0:
			tol += 1
			print("changed tolerance.")
			dist_list = searching_neighborhood(priority_district, tol, dim, lat, lon)
			print(len(dist_list), "distlist length")

		#print(len(dist_list))
		heapq.heapify(dist_list)

		add_block = dist_list[0]
		priority_district.add_block(add_block[1:-2], Districts)
		#print("i j:", add_block[5], add_block[6])
		Grid[int(add_block[5])][int(add_block[6])].remove(add_block[1:-2])
		unassigned_blocks -= 1

		if unassigned_blocks == (data.shape[0] - global_epsilon):
			break
		#print(unassigned_blocks)
		#print("population of priority district", priority_district.population)
		#print("which district", priority_district.id)
	graph(Districts, data)

def graph(Districts, data):
	plt.scatter(data[:, 2], data[:, 1], color='k')

	colors = itertools.cycle(["b", "g", "r", "c", "m", "y"])
	for district in Districts:
		#print("change district")
		#print("blocks in district", len(district.blocks))
		#print("population in district", district.population)
		c = next(colors)
		for block in district.blocks:
			plt.scatter(block[2], block[1], color=c)

	xx = []
	yy = []
	for c in centroid_l:
		xx.append(c[2])
		yy.append(c[1])

	plt.scatter(xx, yy, color='w')#, marker='o')
	plt.savefig(str(global_epsilon)+".png")

	'''
	x_size = (lon[1] - lon[0]) / dim[0]
	y_size = (lat[1] - lat[0]) / dim[1]

	for r in range(dim[0]):
		
		loc = lon[0] - r*y_size

		plt.axvline(x=loc)
	for c in range(dim[1]):
		loc = lat[0] + c*x_size
		
		plt.axhline(y=loc)
	'''

	plt.show()

#data = np.genfromtxt("IL.csv", delimiter=',', skip_header=True)
#plt.scatter(data[:, 2], data[:, 1])
#plt.show()
#plt.savefig("raw.png")


searching_all(filename)
#graph(Districts)