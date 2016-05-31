import district_class as dc
import numpy as np
import math
from grid import build_grid, hash_map_index, grid_is_valid, find_random_centroids
#from centroids import find_random_centroids
import heapq 
import matplotlib.pyplot as plt
import itertools
import sys
import boto3

s3 = boto3.resource('s3') 

class MRStates(MRJob):
    def mapper(self, _, line):
        line = line.split(',')
        print(line[0])
        redistrict(str(line[0]), int(line[1]))

def redistrict(filename, number):
    centroid_l = find_random_centroids(filename, number)
    districts = create_districts(centroid_l)
    statename = (filename[-6:])[0:2]
    searching_all(filename, number, centroid_l, statename)

def euclidean_norm(centroid, block):
    t1 = (centroid[1] - block[1])
    t2 = (centroid[2] - block[2])
    distance = math.sqrt(t1*t1 + t2*t2)
    return distance

def debug(i_0, j_0, Grid, centroid):
    dist_list = []
    for block in Grid[int(i_0)][int(j_0)]:
        dist = euclidean_norm(centroid, block)
        dist_list.append([dist, block[0], block[1], block[2], block[3], i_0, j_0])
    heapq.heapify(dist_list)
    
def neighborhood_to_search(priority_district, dim, lat, lon):
#def neighborhood_to_search(centroid, tol, dim, lat, lon, Grid):
    centroid = priority_district.centroid
    tol = priority_district.tolerance
    i_0, j_0 = hash_map_index(dim, lat, lon, centroid)
    return [max(i_0-tol, 0), min(i_0+tol, dim[1]-1)], [max(j_0-tol, 0), min(j_0+tol, dim[0]-1)]

def search_cell(Grid, i, j, dist_list, priority_district):
    for block in Grid[i][j]:
        dist = euclidean_norm(priority_district.centroid, block)
        dist_list.append([dist, block[0], block[1], block[2], block[3], i, j])

def searching_neighborhood(priority_district, Grid, dim, lat, lon):
    x_range, y_range = neighborhood_to_search(priority_district, dim, lat, lon)
    
    dist_list = []
    for i in range(x_range[0], x_range[1]+1):
        for j in range(y_range[0], y_range[1]+1):
            search_cell(Grid, i, j, dist_list, priority_district)

    return dist_list

def searching_all(filename, number):
    Grid, data, dim, lat, lon = build_grid(filename, number)
    Districts = dc.create_districts(centroid_l, Grid, dim, lat, lon)
    colors_dict = get_colors(Districts)

    unassigned_blocks = data.shape[0]

    while unassigned_blocks != 0:
        priority_district = dc.return_low_pop(Districts)

        while priority_district.neighborhood.shape[0] == 0:
            priority_district.increment_tolerance(Grid, dim, lat, lon)

        add_block = priority_district.neighborhood[0].tolist()
        priority_district.remove_neighborhood_block(add_block)

        if add_block[1:-2] not in Grid[int(add_block[5])][int(add_block[6])]:
            heapq.heappush(Districts, priority_district)
            continue
            
        priority_district.add_block(add_block[1:-2], Districts)

        Grid[int(add_block[5])][int(add_block[6])].remove(add_block[1:-2])

        plt.scatter(add_block[3], add_block[2], color=colors_dict[priority_district.id])
        unassigned_blocks -= 1

    graph(Districts)

def get_colors(Districts):
    colors_dict = {}
    colors = itertools.cycle(["b", "g", "r", "c", "m", "y"])
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
    data = get_data_from_s3(filename)

    CB_Per_GB = (data.shape[0]/number)*(2/9)
    eps = 0.00000001
    max_id, max_lat, max_lon, pop = data.max(axis=0)
    min_id, min_lat, min_lon, min_pop = data.min(axis=0)
    max_lon += eps
    max_lat += eps
    min_lat -= eps
    min_lon -= eps
    blocks = data.shape[0]/CB_Per_GB
    lon_to_lat =  (max_lon - min_lon) / (max_lat - min_lat)
    y_num = math.sqrt(blocks/lon_to_lat)
    x_num = blocks/y_num
    return [int(math.ceil(x_num)), int(math.ceil(y_num))], [min_lat, max_lat], [min_lon, max_lon], data

def hash_map_index(dim, lat, lon, block):
    x_size = (lon[1] - lon[0]) / dim[0]
    y_size = (lat[1] - lat[0]) / dim[1]
    _j = int((float(block[2]) - lon[0]) / x_size) 
    _i = int((float(block[1]) - lat[0]) / y_size) 
    j = (dim[0]-1) - _j
    i = (dim[1]-1) - _i
    return i, j

def find_random_centroids(filename, number):
    '''
    Inputs: filename, number of districts.
    Outputs: a list of centroids, which are described by
    unique id, latitude, longitude and population. 

    Choice of centroids is random. 
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
    #if not grid_is_valid(dim, lat, lon, Master_Grid):
    #    return
    return Master_Grid, data, dim, lat, lon

class District:
    def __init__(self, centroid, district_id, Grid, dim, lat, lon):
        self.blocks = []
        self.id = district_id
        self.centroid = [centroid[0], centroid[1], centroid[2], centroid[3]]
        self.population = centroid[3]
        self.tolerance = 1
        self.assign_neighborhood(Grid, dim, lat, lon)

    def increment_tolerance(self, Grid, dim, lat, lon):
        self.tolerance += 1
        self.assign_neighborhood(Grid, dim, lat, lon)

    def assign_neighborhood(self, Grid, dim, lat, lon):
        neighborhood = np.array(searching_neighborhood(self, Grid, dim, lat, lon))
        neighborhood = neighborhood[neighborhood[:,0].argsort()]
        self.neighborhood = neighborhood

    def remove_neighborhood_block(self, add_block):
        self.neighborhood = np.delete(self.neighborhood, 0, 0)

    def add_block(self, block, district_list):
        self.blocks.append(block)
        self.population += block[3]
        heapq.heappush(district_list, self)

    def return_population(self):
        return self.population

    def __lt__(self, other):
        return self.population < other.population

    def __repr__(self):
        return str(self.centroid)
        #output must be string


def create_districts(centroid_info, Grid, dim, lat, lon):
    districts = []
    i = 0
    for c in centroid_info:
        new_district = District(c, i, Grid, dim, lat, lon)
        districts.append(new_district)
        i+=1        

    heapq.heapify(districts)
    return districts 

def return_low_pop(districts):
    return heapq.heappop(districts)

if __name__ == '__main__':
    MRStates.run()

'''
if __name__ == "__main__":
    if int(sys.argv[2]) <= 1:
        print("Not enough number of districts.")
        sys.exit(1)

    CENTROID_L = find_random_centroids(sys.argv[1], int(sys.argv[2]))
    #DISTRICTS = dc.create_districts(CENTROID_L)
    #EPSILON = int(sys.argv[3])
    searching_all(sys.argv[1], int(sys.argv[2]))
'''