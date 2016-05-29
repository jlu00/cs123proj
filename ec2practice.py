#from mrjob.job import MRJob
import numpy as np
import heapq
import random
import csv
import math
import matplotlib.pyplot as plt
import itertools
import io
import urllib, base64
import boto3

s3 = boto3.resource('s3') 

#class MRStates(MRJob):
#    def mapper(self, _, line):

#        redistrict(str(stateline[0]), int(stateline[1]))

def redistrict(filename, number):
    statename = str(filename[0:2])
    centroid_l = find_random_centroids(filename, number, statename)
    districts = create_districts(centroid_l)
    
    print(statename)

    searching_all(filename, number, centroid_l, statename)

def euclidean_norm(centroid, block):
    t1 = (centroid[1] - block[1])
    t2 = (centroid[2] - block[2])
    distance = math.sqrt(t1*t1 + t2*t2)
    return distance

def neighborhood_to_search(centroid, tol, dim, lat, lon, Grid):
	i_0, j_0 = hash_map_index(dim, lat, lon, centroid)
	return [max(i_0-tol, 0), min(i_0+tol, dim[1]-1)], [max(j_0-tol, 0), min(j_0+tol, dim[0]-1)]

def searching_neighborhood(priority_district, tol, Grid, dim, lat, lon):
	x_range, y_range = neighborhood_to_search(priority_district.centroid, tol, dim, lat, lon, Grid)
	count = 0
	dist_list = []
	for i in range(x_range[0], x_range[1]+1):
 		for j in range(y_range[0], y_range[1]+1):
 			for block in Grid[i][j]:
 				count += 1
 				dist = euclidean_norm(priority_district.centroid, block)
 				dist_list.append([dist, block[0], block[1], block[2], block[3], i, j])
	return dist_list

def searching_all(filename, number, centroid_l, statename):
    Grid, data, dim, lat, lon = build_grid(filename, number)
    Districts = create_districts(centroid_l)
    unassigned_blocks = data.shape[0]
    colors_dict = get_colors(Districts)
    while unassigned_blocks != 0:
        tol = 1
        priority_district = return_low_pop(Districts)
        dist_list = searching_neighborhood(priority_district, tol, Grid, dim, lat, lon)
        while len(dist_list) == 0:
            tol += 1
            dist_list = searching_neighborhood(priority_district, tol, Grid, dim, lat, lon)
        add_block = min(dist_list)
        priority_district.add_block(add_block[1:-2], Districts)
        Grid[int(add_block[5])][int(add_block[6])].remove(add_block[1:-2])
        
        '''
        The two lines of code below are to make the CSV. 
        '''
        #block_info = [add_block[2], add_block[3], priority_district.id]
        #new_districts_list.append(block_info)
        plt.scatter(add_block[3], add_block[2], color=colors_dict[priority_district.id], s=2)
        if unassigned_blocks == (data.shape[0] - 2500):
            graph(Districts, data, centroid_l, statename)
            break
        if unassigned_blocks%100==0:
            print(unassigned_blocks)
        unassigned_blocks -= 1
       #create_csv(new_districts_list, statename)
    #graph(Districts, data, centroid_l, statename)
'''
def create_csv(new_districts_list, statename):
    with open("/home/student/cs123proj/" + statename + 'districts.csv', 'w') as dictfile:
        dwriter = csv.writer(dictfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for item in new_districts_list:
            dwriter.writerow(item)
    dictfile.close()
'''
def get_colors(Districts):
    colors_dict = {}
    colormap = plt.cm.Accent
    colors = itertools.cycle([colormap(i) for i in np.linspace(0, 0.9, len(Districts))])
    for district in Districts:
        c = next(colors)
        colors_dict[district.id] = c
    return colors_dict


def graph(districts, data, centroid_l, statename):
	#plt.scatter(data[:, 2], data[:, 1], color='k')
    #im = io.BytesIO()
    xx = []
    yy = []
    for c in centroid_l:
        xx.append(c[2])
        yy.append(c[1])

    pic_file = str(statename) + ".png"
    plt.scatter(xx, yy, color='k', s=6)
    plt.savefig(statename+".png")
    
    plt.clf()
    #im.seek(0)
    #imagedata = base64.b64encode(im.read())
    #print(imagedata, "hello")
    imagepath = "/home/ec2-user/" + statename + ".png"
    s3.Object(bucket_name='jun9242.spr16.cs123.uchicago.edu', key=pic_file).put(Body=open(imagepath, 'rb'))

def get_data_from_s3(filename):
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

    #data = np.genfromtxt(filename, delimiter=',', skip_header=True)

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

def find_random_centroids(filename, number, statename):
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
    '''
    with open("/home/student/cs123proj/" + statename + 'centroids.csv', 'w') as cfile:
        cwriter = csv.writer(cfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for c in centroids:
            cwriter.writerow(c)
    cfile.close()
    '''
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

def create_districts(centroid_info):
    districts = []
    i = 0
    for c in centroid_info:
        new_district = District(c, i)
        districts.append(new_district)
        i+=1        

    heapq.heapify(districts)
    return districts 

def return_low_pop(districts):
    return heapq.heappop(districts)

redistrict('HI.csv', 2)
#redistrict('NH.csv', 2)