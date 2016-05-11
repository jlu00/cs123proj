import math
#3500 census blocks per grid block
CB_Per_GB = 3500

class Grid_Block():
	def __init__(self, ij_tuple):
		self.id = ij_tuple
		self.blocks = []

	def add_block(self, block):
		self.blocks.append(block)


def create_grid(filename):
	'''
	[ (id, lat, long, pop),
	   (id, lat, long, pop)]

	[ [id, lat, long, pop],
	  [id, lat, long, pop] ]
	'''
	data = np.genfromtxt(filename, skip_header=True)
	lat = np.sort()
	max_lat = lat[0]
	min_lat = lat[-1]
	lon = np.sort()
	max_lon = lon[0]
	min_lon = lon[-1]
	blocks = data.shape[0]/CB_Per_GB
	lat_to_lon = (max_lat - min_lat) / (max_lon - min_lon)
	y_num = math.ceiling(blocks/(lat_to_lon + 1))
	x_num = math.ceiling(lat_to_lon*(y_dim))

	Master_Grid = []
	for r in range(y_num):
		for c in range(x_num):
			Master_Grid.append(Grid_Block((r,c)))
	return Master_Grid

def build_grid(data, Master_Grid):
	for item in data:
		(i, j) = hash_map_index(item)
		for block in Master_Grid:
			if block.id == (i, j):
				block.add_block(item)

def hash_map_index(x_num, y_num, max_lat, min_lat, max_lon, min_lon, block):
	x_size = (max_lat - min_lat) / x_num
	y_size = (max_lat - min_lat) / y_num

	_j = math.ceiling((block[1] - min_lat) / x_size) - 1
	_i = math.ceiling((block[2] - min_lon) / y_size) - 1

	j = x_num - _j
	i = y_num - _i
	return (i, j)
