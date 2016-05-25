import sys

def parse_cluster_data(header, pop, results):
	with open(header, 'r') as header_file:
		with open(pop, 'r') as pop_file: 
			f = open(results, 'w+')
			f.write('block_num,latitude,longitude,total_pop'+'\n')
			header_line = 'header'
			block_num = 1
			while header_line != '':
				header_line = header_file.readline()
				pop_line = pop_file.readline()
				if header_line[8:11]=='101':
					lat = header_line[310:313] + '.' + header_line[313:319]
					lon = header_line[319:323] + '.' + header_line[323:329]
					pop_line = pop_line.split(',')
					total_pop = pop_line[5]
					data = '%d' % block_num +','+ lat +','+ lon +','+ total_pop +'\n'
					block_num+=1
					f.write(data)


	f.close()

if __name__ == "__main__":
	parse_cluster_data('data/' + sys.argv[1] + 'geo.uf1', 'data/' + sys.argv[1] + '00001.uf1',  sys.argv[1].upper() + '.csv')				 
