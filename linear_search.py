import district_class as dc
import numpy as np
import math

data = np.genfromtxt('IL.csv', delimiter=',', skip_header=True)
print(data)

centroid_l = [[1032,-089.192249,41.326240,39],
[1023,-089.143829,38.530543,18],
[3001,-087.693154,41.897982,374],
[3023,-088.250446,41.992887,51],
[1010,-087.713445,41.510284,6],
[4035,-089.048809,40.119928,1],
[2007,-087.679184,39.628681,23],
[1018,-088.423013,41.367358,90],
[1044,-088.792693,38.259951,17],
[2031,-088.332532,41.543430,5],
[1054,-089.110784,41.542103,26],
[2005,-088.443526,42.335221,54],
[5034,-090.055654,38.823547,28],
[1108,-090.301684,38.296993,4],
[1031,-088.736112,40.014611,0],
[2050,-089.921620,38.578698,38],
[1046,-089.792060,42.490470,39],
[3146,-088.086253,38.140670,4],
[3061,-089.022686,37.722950,0]]

Districts = dc.create_districts(centroid_l, 1)
init_idx = np.where(data == 1031)
print(data[init_idx[0][0]])

def euclidean_norm(centroid, block):
	distance = math.sqrt((centroid[0]-block[1])**2+(centroid[1]-block[2])**2)
	return distance

while data.shape[0] != 0:	
	#print('start while, data shape is:', data.shape[0])	
 	priority_district = dc.return_low_pop(Districts)
 	#print("centroid loc", priority_district.centroid[0], priority_district.centroid[1])

	dtype = [('distance', float), ('id', int), ('pop', int)]
	dist_list = np.empty(data.shape[0], dtype = dtype)
	#print('dist_list shape:', dist_list.shape)

	for i in range(data.shape[0]):
		dist = euclidean_norm(priority_district.centroid, data[i][:])
		dist_list[i] = (dist, data[i][0], data[i][3])

	#print("unsorted", dist_list[0])
	dist_list = np.sort(dist_list, order='distance')
	priority_district.add_block(dist_list[0])
	#print("sorted", dist_list[0][1])
	idx = np.where(data == dist_list[0][1])#[0][0]
	#print('idx is', idx)
	#print('data idx', data[idx[0][0]])
	data = np.delete(data, idx[0][0], 0)




