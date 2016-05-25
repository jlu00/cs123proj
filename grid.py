import csv
import random
import math
import numpy as np

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

	#print("Create Grid")
	#print(max_lat, max_lon)
	#print(min_lat, min_lon)

	blocks = data.shape[0]/CB_Per_GB
	#print(blocks)
	lon_to_lat =  (max_lon - min_lon) / (max_lat - min_lat) #cannot be wrong
	#print(lon_to_lat)
	y_num = math.sqrt(blocks/lon_to_lat)
	x_num = blocks/y_num
	#print(x_num, y_num)

	return [int(math.ceil(x_num)), int(math.ceil(y_num))], [min_lat, max_lat], [min_lon, max_lon], data


def hash_map_index(dim, lat, lon, block):
	x_size = (lon[1] - lon[0]) / dim[0]
	y_size = (lat[1] - lat[0]) / dim[1]
	#print("x_size", x_size)
	#print("y_size", y_size)

	_j = int((float(block[2]) - lon[0]) / x_size) 
	_i = int((float(block[1]) - lat[0]) / y_size) 
	#print("_i", _i)
	#print("_j", _j)
	
	j = (dim[0]-1) - _j
	i = (dim[1]-1) - _i
	#print("i", i)
	#print("j", j)
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
					print("i, j", i, j)
					print("x, y", x, y)
					print("block", block)
					print("lat lon", lat, lon)
					print("dim")
					print("Wrong i")
					return False
				if y != j:
					print("i, j", i, j)
					print("x, y", x, y)
					print("block", block)
					print("lat lon", lat, lon)
					print("dim")
					print("Wrong j")
					return False
	print("grid is valid counted blocks", count)
	return True

def build_grid(filename, number):
	dim, lat, lon, data = create_grid(filename, number)

	Master_Grid = []
	for r in range(dim[1]):
		#print([[]]*dim[0])
		row = []
		for c in range(dim[0]):
			row.append([])
		#Master_Grid.append([[]]*dim[0])
		Master_Grid.append(row)

	#print(Master_Grid)
	#print("data shape", data.shape[0])
	count = 0
	for item in data:
		count += 1
		i, j = hash_map_index(dim, lat, lon, item)
		#print("i j", i, j)
		#print(Master_Grid[i][j])
		Master_Grid[i][j].append(item.tolist())
		#print(Master_Grid[i][j])
		#print(Master_Grid)
	if not grid_is_valid(dim, lat, lon, Master_Grid):
		return

	#print("counted blocks", count)
	print("Built grid", len(Master_Grid), len(Master_Grid[0]))
	return Master_Grid, data, dim, lat, lon
