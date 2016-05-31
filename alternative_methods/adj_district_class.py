import heapq
from adj_grid_search import searching_neighborhood
import numpy as np

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












