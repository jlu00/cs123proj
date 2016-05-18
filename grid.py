import math
import numpy as np
#3500 census blocks per grid block
CB_Per_GB = 3500


def create_grid(filename):
	'''
	[ (id, lat, long, pop),
	   (id, lat, long, pop)]

	[ [id, lat, long, pop],
	  [id, lat, long, pop] ]
	'''
	data = np.genfromtxt(filename, delimiter=',', skip_header=True)

	max_id, max_lat, max_lon, pop = data.max(axis=0)
	min_id, min_lat, min_lon, min_pop = data.min(axis=0)

	blocks = data.shape[0]/CB_Per_GB
	lon_to_lat =  (max_lon - min_lon) / (max_lat - min_lat)
	x_num = int(math.sqrt(blocks/lon_to_lat))
	y_num = int(math.ceil(blocks/x_num))

	return [x_num+1, y_num], [min_lat, max_lat], [min_lon, max_lon], data


def hash_map_index(dim, lat, lon, block):
	x_size = (lon[1] - lon[0]) / dim[0]
	y_size = (lat[1] - lat[0]) / dim[1]

	_j = int((block[1] - lat[0]) / y_size)
	_i = int((block[2] - lon[0]) / x_size)
	
	j = (dim[0]-1) - _i
	i = (dim[1]-1) - _j
	return i, j
	
def build_grid(filename):
	dim, lat, lon, data = create_grid(filename)

	Master_Grid = []
	for r in range(dim[1]):
		Master_Grid.append([[]]*dim[0])

	#for c in range(dim[0]):
	#	Master_Grid.append([[]]*dim[1])
		
	for item in data:
		i, j = hash_map_index(dim, lat, lon, item)
		Master_Grid[i][j].append(item.tolist())

	print("Built grid", len(Master_Grid), len(Master_Grid[0]))
	return Master_Grid, data, dim, lat, lon
