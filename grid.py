import math
import numpy as np
#3500 census blocks per grid block
CB_Per_GB = 3500

'''
class Grid_Block():
	def __init__(self, ij_tuple):
		self.id = ij_tuple
		self.blocks = []

	def add_block(self, block):
		self.blocks.append(block)
'''

def create_grid(filename):
	'''
	[ (id, lat, long, pop),
	   (id, lat, long, pop)]

	[ [id, lat, long, pop],
	  [id, lat, long, pop] ]
	'''
	data = np.genfromtxt(filename, delimiter=',', skip_header=True)
	#dtype = [('distance', float), ('id', int), ('pop', int)]

	#lat = np.sort(dist_list, order='distance')
	max_id, max_lat, max_lon, pop = data.max(axis=0)
	min_id, min_lat, min_lon, min_pop = data.min(axis=0)

	#max_lat = lat[0]
	#min_lat = lat[-1]
	#lon = np.sort()
	#max_lon = lon[0]
	#min_lon = lon[-1]
	#print("max lat, lon", max_lat, max_lon)
	#print("min lat, lon", min_lat, min_lon)
	blocks = data.shape[0]/CB_Per_GB
	#print("blocks", blocks)
	lat_to_lon =  (max_lon - min_lon) / (max_lat - min_lat)
	#print("R", lat_to_lon)
	x_num = int(math.sqrt(blocks/lat_to_lon))
	y_num = int(math.ceil(blocks/x_num))

	#y_num = int(math.ceil(blocks/(lat_to_lon + 1)))
	#x_num = int(math.ceil(lat_to_lon*(y_num)))
	#print("x dimensions:", x_num)
	#print("y_dimensions:", y_num)

	return x_num+1, y_num, max_lat, min_lat, max_lon, min_lon, data


def hash_map_index(x_num, y_num, max_lat, min_lat, max_lon, min_lon, block):
	x_size = (max_lon - min_lon) / x_num
	y_size = (max_lat - min_lat) / y_num

	_j = int((block[1] - min_lat) / y_size)
	_i = int((block[2] - min_lon) / x_size)
	#print("_i", _i)
	#print("_j", _j)

	#print("x_num", x_num)
	#print("y_num", y_num)
	j = (x_num-1) - _i
	i = (y_num-1) - _j
	print("block", block)
	print("i", i)
	print("j", j)
	return i, j

def build_grid(filename):
	x_num, y_num, max_lat, min_lat, max_lon, min_lon, data = create_grid(filename)

	Master_Grid = []
	for r in range(y_num):
		Master_Grid.append([[]]*x_num)
		#Master_Grid[r] = []*x_num
		#for c in range(x_num):
			#Master_Grid[r].append([])
			#print(Master_Grid[r][c])

	for item in data:
		#print('item', item)
		i, j = hash_map_index(x_num, y_num, max_lat, min_lat, max_lon, min_lon, item)
		#print("max lat, lon", max_lat, max_lon)
		#print("min lat, lon", min_lat, min_lon)
		#print(type(Master_Grid[i]), "hello")
		#print(type(Master_Grid[i][j]))
		#Master_Grid[i][j] = []
		#print("hi")
		Master_Grid[i][j].append(item)

	print(Master_Grid)


build_grid('IL.csv')


