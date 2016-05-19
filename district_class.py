import heapq

class District:
	def __init__(self, centroid, tolerance, district_id):
		self.blocks = []
		self.id = district_id
		self.centroid = [centroid[0], centroid[1], centroid[2], centroid[3]]
		self.centroid_id = centroid[0]
		self.centroid_lat = centroid[1]
		self.centroid_lon = centroid[2]
		self.centroid_pop = centroid[3]
		self.population = centroid[3]
		self.tolerance = tolerance

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

def create_districts(centroid_info, tolerance):
	districts = []
	i = 0
	for c in centroid_info:
		new_district = District(c, tolerance, i)
		districts.append(new_district)
		i+=1		

	heapq.heapify(districts)
	return districts 

def return_low_pop(districts):
	return heapq.heappop(districts)












