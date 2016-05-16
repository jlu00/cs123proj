import district_class as dc
import numpy as np
import math
from grid.py import build_grid 
import heapq 

centroid_l = [[1032,-089.192249,41.326240,39],
[1023,-089.143829,38.530543,18],
[3001,-087.693154,41.897982,374],
[3023,-088.250446,41.992887,51],
[1010,-087.713445,41.510284,6],
[4035,-089.048809,40.119928,1],
[2007,-087.679184,39.628681,23],
[1018,-088.423013,41.367358,90],
[1044,-088.792693,38.259951,17],
[2031,-088.332532,41.543430,5],
[1054,-089.110784,41.542103,26],
[2005,-088.443526,42.335221,54],
[5034,-090.055654,38.823547,28],
[1108,-090.301684,38.296993,4],
[1031,-088.736112,40.014611,0],
[2050,-089.921620,38.578698,38],
[1046,-089.792060,42.490470,39],
[3146,-088.086253,38.140670,4],
[3061,-089.022686,37.722950,0]]

Districts = dc.create_districts(centroid_l, 1)
Grid, data, dim, lat, lon = build_grid(filename)

def euclidean_norm(centroid, block):
	distance = math.sqrt((centroid[0]-block[1])**2+(centroid[1]-block[2])**2)
	return distance

def neighborhood_to_search(centroid, tol):
	i_0, j_0 = hash_map_index(dim, lat, lon, centroid)
	return [max(i_0-tol, 0), min(i_0+tol, len(Grid[0])], [max(j_0-tol, 0), min(j_0+tol, len(Grid)])

def searching_neighborhood(priority_district, tol):
	y_range, x_range = search_neighborhood(priority_district, tol)
 	dist_list = []
 	for i in x_range:
 		for j in y_range:
 			for block in Grid[i][j]:
				dist = euclidean_norm(priority_district.centroid, data[i][:])
				dist_list.append((dist, data[i][0], data[i][1], data[i][2], data[i][3]))
	return dist_list

def searching_all(filename):
	unassigned_blocks = data.shape[0]
	while unassigned_blocks != 0:
		tol = 1
	 	priority_district = dc.return_low_pop(Districts)
	 	dist_list = searching_neighborhood(priority_district, tol)
		
		while dist_list == 0:
			tol += 1
			dist_list = searching_neighborhood(priority_district, tol)

		heapq.heapify(dist_list)
		priority_district.add_block(dist_list[0])
		unassigned_blocks -= 1

searching_all("IL.csv")