from mrjob.job import MRJob
from mrjob.step import MRStep
import mrjob
import re



class MRClosestBlock(MRJob):
	def mapper(self, _, line):
		pass
	def combiner(self, _, line):
		pass
	def reducer(self, _, line):
		pass



if __name__ == '__main__':
	MRClosestBlock.run()