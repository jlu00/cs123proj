from mrjob.job import MRjob
from mrjob.step import MRStep
import mrjob
import grid_search.py
import matplotlib.pyplot as plt
import os


""

class MRStates(MRJob):
	OUTPUT_PROTOCOL = mrjob.protocol.JSONValueProtocol
	def mapper(self, _, line):
		grid_search(line[0], line[1])
        fields = line.split(",")
        fields[0] fields[1]
        yield

if __name__ = '__main__':
	MRStates.run()