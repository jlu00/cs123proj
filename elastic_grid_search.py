from mrjob.job import MRJob
import numpy as np
import heapq
import random
import csv
import math
import matplotlib.pyplot as plt
import itertools
import boto3

s3 = boto3.resource('s3') 

class MRStates(MRJob):
    '''
    Inputs: the line which has the STATE.csv filename
    and the number of districts.
    '''
    def mapper(self, _, line):
        line = line.split(',')
        redistrict(str(line[0]), int(line[1]))

def redistrict(filename, number):
    '''
    Inputs: filename and number of districts
    Outputs: None

    Calls the find_random_centroids function to make a list of random centroids. 
    Calls the create_districts function to create district classes from the centroid
    list. Calls searching_all function to redistrict the state.
    '''
    centroid_l = find_random_centroids(filename, number)
    districts = create_districts(centroid_l)
    statename = filename[0:2]
    searching_all(filename, number, centroid_l, statename)

def euclidean_norm(centroid, block):
    '''
    Inputs: centroid of district (list) and census block (list) being evaluated. 
    Outputs: returns the euclidean difference (float) between the two.
    '''
    t1 = (centroid[1] - block[1])
    t2 = (centroid[2] - block[2])
    distance = math.sqrt(t1*t1 + t2*t2)

    return distance

def neighborhood_to_search(centroid, tol, dim, lat, lon, Grid):
    '''
    Inputs: centroid of district (list), tol (int), dimensions of Grid
            (list of ints), max and min latitudes (list of ints), max and min longitudes
            (list of ints), the Grid (nested lists of lists containing lists of blocks)
    Outputs: array of i values to search, array of j values to search
    In the case of a centroid being located in a border cell, i and j values are 
    truncated to not extend past the grid.
    '''
	i_0, j_0 = hash_map_index(dim, lat, lon, centroid) #Grid cell containing centroid

	return [max(i_0-tol, 0), min(i_0+tol, dim[1]-1)], [max(j_0-tol, 0), min(j_0+tol, dim[0]-1)]

def searching_neighborhood(priority_district, tol, Grid, dim, lat, lon):
    '''
    Inputs: priority_district object, tol (int), Grid (nested lists of lists containing 
            lists of blocks), dimensions of Grid (list of ints), max and min latitudes 
            (list of ints), max and min longitudes (list of ints)
    Outputs: dist_list list of lists- each item is a block with distance to centroid,
            block id, latitude, longitude, population, and i and j of Grid location
    Iterates through every block in the cells yielded by every combination of i and j 
    as determined by neighborhood_to_search, to return dist_list.
    '''
	x_range, y_range = neighborhood_to_search(priority_district.centroid, tol, dim, lat, lon, Grid)
	dist_list = []

	for i in range(x_range[0], x_range[1]+1):
 		for j in range(y_range[0], y_range[1]+1):
 			for block in Grid[i][j]:
 				dist = euclidean_norm(priority_district.centroid, block)
 				dist_list.append([dist, block[0], block[1], block[2], block[3], i, j])

	return dist_list

def searching_all(filename, number, centroid_l, statename):
    '''
    Inputs: filename (string) raw data file, number (int) number of
            districts, centroid_l (list of lists) list of centroids, statename (string)
            state to search
    Iterates through all unassigned blocks, and assigns to highest priority district.
    '''
    #Build Grid
    Grid, data, dim, lat, lon = build_grid(filename, number)
    plt.scatter(data[:, 2], data[:, 1], color='k')

    #Initialize list of districts
    Districts = create_districts(centroid_l)
    colors_dict = get_colors(Districts)

    #Iterate through all blocks in the state
    unassigned_blocks = data.shape[0]
    while unassigned_blocks != 0:
        tol = 1
        priority_district = return_low_pop(Districts)
        dist_list = searching_neighborhood(priority_district, tol, Grid, dim, lat, lon)

        #Increment tolerance if immediate neighborhood is empty
        while len(dist_list) == 0:
            tol += 1
            dist_list = searching_neighborhood(priority_district, tol, Grid, dim, lat, lon)

        #Find nearest block, delete from Grid, and plot
        add_block = min(dist_list)
        priority_district.add_block(add_block[1:-2], Districts)
        Grid[int(add_block[5])][int(add_block[6])].remove(add_block[1:-2])
        plt.scatter(add_block[3], add_block[2], 
                    color=colors_dict[priority_district.id], s=2)
        
        #Partial Plot code for debugging:
        #if unassigned_blocks == (data.shape[0] - epsilon):
        #   break

        unassigned_blocks -= 1

    graph(Districts, centroid_l, statename) #graphs the centroids 

def get_colors(Districts):
    '''
    Inputs: a list of district classes
    Outputs: colors_dict, a dictionary with district id 
            as the key and the color code from the 
            Accent color map library as the value. 
    '''
    colors_dict = {}
    colormap = plt.cm.Accent
    colors = itertools.cycle([colormap(i) for i in np.linspace(0, 0.9, len(Districts))])

    for district in Districts:
        c = next(colors)
        colors_dict[district.id] = c

    return colors_dict

def graph(districts, centroid_l, statename):
    '''
    Inputs: list of district classes, the list of centroids, the name of the state.
    Outputs: Saves a png of the graphed result into the s3Bucket. 
    '''
    xx = []
    yy = []
    for c in centroid_l:
        xx.append(c[2])
        yy.append(c[1])

    pic_file = str(statename) + ".png"
    plt.scatter(xx, yy, color='w', s=6)
    plt.savefig(statename+".png")
    plt.clf()
    imagepath = statename + ".png"
    s3.Object(bucket_name='jun9242.spr16.cs123.uchicago.edu', key=pic_file).put(Body=open(imagepath, 'rb'))

def get_data_from_s3(filename):
    '''
    Input: filename
    Output: Data in a numpy array of shape (number of censusblocks, 4)
    Where the 4 fields are unique id, latitude, longitude and population
    '''
    info = s3.Object(bucket_name='jun9242.spr16.cs123.uchicago.edu', key=filename).get()
    chunk = info["Body"].read()
    chunk_string = chunk.decode("utf-8")
    data = chunk_string.split()

    for i in range(len(data)):
        data[i] = data[i].split(',')

    data = np.asarray(data[1:], dtype=float, order='F')

    return data

def create_grid(filename, number):
    '''
    Inputs: The name of the file and the number of districts.
    Outputs: array of Grid dimensions, array of min and max latitude, array of
             min and max longitude, data numpy array of blocks
    Calculates details of grid to be built from raw data.
    '''
    data = get_data_from_s3(filename)

    #Census blocks per grid cell- estimated s.t. 9 cells will have twice the 
    #number of census blocks per district, assuming equal distribution of blocks
    #per district
    CB_Per_GB = (data.shape[0]/number)*(2/9)
    eps = 0.00000001
    max_id, max_lat, max_lon, pop = data.max(axis=0)
    min_id, min_lat, min_lon, min_pop = data.min(axis=0)

    #Handle endpoints by adding eps
    max_lon += eps
    max_lat += eps
    min_lat -= eps
    min_lon -= eps
    blocks = data.shape[0]/CB_Per_GB

    #Ratio of longitude to latitude
    lon_to_lat =  (max_lon - min_lon) / (max_lat - min_lat)

    #Calculate dimensions of grid while roughly preserving the ratio of
    #state's longitude to latitute
    y_num = math.sqrt(blocks/lon_to_lat)
    x_num = blocks/y_num

    return [int(math.ceil(x_num)), int(math.ceil(y_num))], [min_lat, max_lat], [min_lon, max_lon], data

def hash_map_index(dim, lat, lon, block):
    '''
    Inputs: dimensions of Grid (list of ints), max and min latitudes (list of ints), 
            max and min longitudes (list of ints), block (list) census block
    Outputs: i (int) index of grid cell on y axis, j (int) index of grid cell on x axis
    Calculates indices of block location in Grid.
    '''
    #Size of grid cell in latitude or longitude degrees
	x_size = (lon[1] - lon[0]) / dim[0]
	y_size = (lat[1] - lat[0]) / dim[1]

    #Index of block in Grid, where (0,0) is origin
	_j = int((float(block[2]) - lon[0]) / x_size) 
	_i = int((float(block[1]) - lat[0]) / y_size) 

    #Index of block in Grid, where (0,0) is upper left
	j = (dim[0]-1) - _j
	i = (dim[1]-1) - _i
	return i, j

def find_random_centroids(filename, number):
    '''
    Inputs: filename, number of districts.
    Outputs: a list of centroids, which are described by
        unique id, latitude, longitude and population. 

    Chooses random centroids. Centroids may not be located in the same grid cell. 
    '''
    random.seed(0)
    hash_list = []
    centroid_list = []
    dim, lat, lon, data = create_grid(filename, number)

    start = 0
    while start < number:
        choice = random.randint(0, len(data)-1)
        random_block = data[choice]
        hm_tuple = hash_map_index(dim, lat, lon, random_block)
        if hm_tuple not in hash_list:
            hash_list.append(hm_tuple)
            centroid_list.append(random_block.tolist())
            start += 1
        centroids = []

    for c in centroid_list:
        formatted_c = []
        for d in c:
            formatted_c.append(float(d))
        centroids.append(formatted_c)

    return centroids


def grid_is_valid(dim, lat, lon, Grid):
    '''
    Inputs: dimensions of Grid (list of ints), max and min latitudes (list of ints), 
            max and min longitudes (list of ints), Grid (nested lists of lists) of 
            census blocks
    Outputs: Boolean
    Returns True if all block entries in Grid are in the right location as 
    calculated by hash_map_index. Used for debugging.
    '''
	count = 0
	for x in range(dim[1]):
		for y in range(dim[0]):
			for block in Grid[x][y]:
				count += 1

				i, j = hash_map_index(dim, lat, lon, block)
				if x != i:
					return False
				if y != j:
					return False

	return True

def build_grid(filename, number):
    '''
    Inputs: filename (string) csv to read raw data from, number (int) of districts needed.
    Outputs: Grid (nested lists of lists) of census blocks, data (np array) of census blocks,
            dimensions of Grid (list of ints), max and min latitudes (list of ints), max and 
            min longitudes (list of ints)
    Create a Grid with dim dimensions, iterate through all items in data, and place in appropriate
    grid cell.
    '''
    dim, lat, lon, data = create_grid(filename, number)
    Master_Grid = []

    for r in range(dim[1]):
        row = []
        for c in range(dim[0]):
            row.append([])
            Master_Grid.append(row)
            count = 0

    for item in data:
        count += 1
        i, j = hash_map_index(dim, lat, lon, item)
        Master_Grid[i][j].append(item.tolist())
   
    return Master_Grid, data, dim, lat, lon

class District:
    '''
    Builds the district class with a centroid coordinate.
    '''
    def __init__(self, centroid, district_id):
        self.blocks = []
        self.id = district_id
        self.centroid = [centroid[0], centroid[1], centroid[2], centroid[3]]
        self.centroid_id = centroid[0]
        self.centroid_lat = centroid[1]
        self.centroid_lon = centroid[2]
        self.centroid_pop = centroid[3]
        self.population = centroid[3]

    def add_block(self, block, district_list):
        '''
        Inputs: the block array and a list of districts

        Adds a block to the district's list of blocks. 
        '''
        self.blocks.append(block)
        self.population += block[3]
        heapq.heappush(district_list, self)

    def return_population(self):
        '''
        Returns the population of the District.
        '''
        return self.population

    def __lt__(self, other):
        return self.population < other.population

    def __repr__(self):
        #output must be string
        return str(self.centroid)

def create_districts(centroid_info):
    '''
    Inputs: The list of centroid arrays
    Outputs: a list of district classes.

    Makes a list of district classes where
    each district uses an element from centroid.
    '''
    districts = []
    i = 0
    for c in centroid_info:
        new_district = District(c, i)
        districts.append(new_district)
        i+=1        
    heapq.heapify(districts)
    
    return districts 

def return_low_pop(districts):
    '''
    Returns the district with the lowest population with heapop.
    '''
    return heapq.heappop(districts)

if __name__ == '__main__':
	MRStates.run()