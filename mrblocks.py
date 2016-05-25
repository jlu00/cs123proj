from mrjob.job import MRjob
from mrjob.step import MRStep
import mrjob
from grid_search import searching_all
from grid_search import graph
import matplotlib.pyplot as plt


class MRStates(MRJob):
	OUTPUT_PROTOCOL = mrjob.protocol.JSONValueProtocol
	def mapper(self, _, line):
		state_filename = line.split(",\n")
		grid_search(state_filename)
		graph(state_filename)

		yield None, None

	def steps(self):
		return [
			MRStep(mapper=self.mapper)
		]

if __name__ = '__main__':
	MRStates.run()