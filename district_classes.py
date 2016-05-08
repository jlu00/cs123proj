
class Current_District:
	def __init__(self, block, district_id, d_pop, block_pop):
		self.district_id = district_id
		self.block = block
		self.weight = block_pop/d_pop


def represent_existing_districts(filename):
	'''
	Takes in a csv file of the districts of the state
	and creates district classes. 
	'''

'''
Parameters: 
Proportion of pop = weight 
Race: 
Age
Gender
Ethnicity
Household types, presence, children, size, etc.


csv
congressional district + important information (20 parameters)

need tables p1, p2, p3 

csv district assumption: 
district_number, total population, urban proportion, white alone, black alone, asian + pacific islander alone,
american indian alone, proportion of 2 races, hispanic and latino proportion, proportion over 18, male proportion,
median age, average household size, proportion living in households, proportion over 65, average family size, 


'''