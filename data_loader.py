import numpy as np
import os
from glob import glob

class DataLoader:
	
	def __init__(self):
	
		self.trainSpecsLoc="./specs/train/*"
		self.trainGTLoc="./gt/train/*"
		self.testSpecsLoc="./specs/test/*"
		self.testGTLoc="./gt/test/*"

		self.trainSpecsFiles = None
		self.trainGTFiles = None
		self.testSpecsFiles = None
		self.testGTFiles = None	

		self.trainSpecs = None
		self.trainGT = None
		self.testSpecs = None
		self.testGT = None

		self.batchIndex = 0	# start at the beginning
		self.testIndex = 0	

	def loadData(self):

		self.trainSpecsFiles = np.array(glob(self.trainSpecsLoc))
		self.trainGTFiles = np.array(glob(self.trainGTLoc))
		self.testSpecsFiles = np.array(glob(self.testSpecsLoc))
		self.testGTFiles = np.array(glob(self.testGTLoc))

		self.trainSpecs = np.stack([np.load(file) for file in self.trainSpecsFiles])
		self.trainGT = np.stack([np.load(file) for file in self.trainGTFiles])
		self.testSpecs = np.stack([np.load(file) for file in self.testSpecsFiles])
		self.testGT = np.stack([np.load(file) for file in self.testGTFiles])		

	def getBatch(self, size, rand=False):

		batchSpecs = None
		batchGT = None
		
		if(rand):	# choose index randomly
			index = np.random.randint(self.trainSpecs.shape[0], size=size)
		else:	# assign index based on batch index
			# indices to grab are from the current batchIndex plus the size
			index = range(self.batchIndex, self.batchIndex + size)
			index = np.remainder(index, self.trainSpecs.shape[0])
			# keep track of where batchIndex should be for next time
			self.batchIndex = (self.batchIndex + size) % self.trainSpecs.shape[0]
		
		batchSpecs = self.trainSpecs[[i for i in index]]
		batchGT = self.trainGT[[i for i in index]]
		
		return batchSpecs, batchGT

	def getTest(self, size, all=False):

		test_specs = None
		test_gt = None
		
		if(all):
			test_specs = self.testSpecs
			test_gt = self.testGT
		else:	# randomly choose
			index = np.random.randint(self.trainSpecs.shape[0], size=size)

		return test_specs, test_gt
		

