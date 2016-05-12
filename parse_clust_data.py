
def parse_cluster_data(header, pop, results):
	with open(header, 'r') as header_file:
		with open(pop, 'r') as pop_file: 
			f = open(results, 'w+')
			f.write('block_num,longitude,latitude,total_pop'+'\n')
			header_line = 'header'
			while header_line != '':
			#for i in range(430857):
				header_line = header_file.readline()
				pop_line = pop_file.readline()
				if header_line[8:11]=='101':
					lat = header_line[310:313] + '.' + header_line[313:319]
					lon = header_line[319:323] + '.' + header_line[323:329]
					block_num = header_line[62:66]
					pop_line = pop_line.split(',')
					total_pop = pop_line[5]
					data = block_num +','+ lon +','+ lat +','+ total_pop +'\n'
					f.write(data)
			'''
			for line in header_file:	
				if line[8:11]=='101':
					lat = line[310:313] + '.' + line[313:319]
					lon = line[319:323] + '.' + line[323:329]
					block_num = line[62:66]
					pop_line = pop_file.readline().split(',')
					total_pop = pop_line[5]
					data = block_num +','+ lon +','+ lat +','+ total_pop +'\n'
					f.write(data)
			'''
	f.close()
				
parse_cluster_data('ilgeo.uf1', 'il00001.uf1', 'IL.csv')				 
