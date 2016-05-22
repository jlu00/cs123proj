import math
import numpy as np
from centroids import find_random_centroids

def create_grid(filename, number):
	'''
	[ (id, lat, long, pop),
	   (id, lat, long, pop)]

	[ [id, lat, long, pop],
	  [id, lat, long, pop] ]
	'''
	data = np.genfromtxt(filename, delimiter=',', skip_header=True)
	CB_Per_GB = (data.shape[0]/number)*(2/9)
	

	max_id, max_lat, max_lon, pop = data.max(axis=0)
	min_id, min_lat, min_lon, min_pop = data.min(axis=0)

	blocks = data.shape[0]/CB_Per_GB
	lon_to_lat =  (max_lon - min_lon) / (max_lat - min_lat) #cannot be wrong

	y_num = math.sqrt(blocks/lon_to_lat)
	x_num = blocks/y_num

	return [int(math.ceil(x_num)), int(math.ceil(y_num))], [min_lat, max_lat], [min_lon, max_lon], data


def hash_map_index(dim, lat, lon, block):
	x_size = (lon[1] - lon[0]) / dim[0]
	y_size = (lat[1] - lat[0]) / dim[1]
	#print("x_size", x_size)
	#print("y_size", y_size)

	_j = int((block[2] - lon[0]) / x_size) 
	_i = int((block[1] - lat[0]) / y_size) 
	#print("_i", _i)
	#print("_j", _j)
	
	
	j = (dim[0]-1) - _j
	i = (dim[1]-1) - _i
	#print("i", i)
	#print("j", j)
	return i, j
	
def build_grid(filename):
	dim, lat, lon, data = create_grid(filename, 1)

	Master_Grid = []
	for c in range(dim[1]):
		print([[]]*dim[0])
		Master_Grid.append([[]]*dim[0])

	for item in data:
		i, j = hash_map_index(dim, lat, lon, item)
		Master_Grid[i][j].append(item.tolist())

	print("Built grid", len(Master_Grid), len(Master_Grid[0]))
	return Master_Grid, data, dim, lat, lon
