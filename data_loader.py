import numpy as np
import os
from glob import glob

class DataLoader:
	
	def __init__(self):

		# naming convention of how training and test data and gts are saved
		self.trainSpecsLoc="./specs/train/*"
		self.trainGTLoc="./gt/train/*"
		self.testSpecsLoc="./specs/test/*"
		self.testGTLoc="./gt/test/*"

		# will store lists of the file names
		self.trainSpecsFiles = None
		self.trainGTFiles = None
		self.testSpecsFiles = None
		self.testGTFiles = None	

		self.trainSpecs = None
		self.trainGT = None
		self.testSpecs = None
		self.testGT = None

		self.batchIndex = 0	# start at the beginning
		self.testIndex = 0	# beginning


	'''
	Name = loadData
	Purpose = to load all of the training and test data before running the classifier
	Parameters = only itself
	Returns = no return
	'''
	def loadData(self):
		# makes np.array of the file names of all training and test data
		self.trainSpecsFiles = np.array(glob(self.trainSpecsLoc))
		self.trainGTFiles = np.array(glob(self.trainGTLoc))
		self.testSpecsFiles = np.array(glob(self.testSpecsLoc))
		self.testGTFiles = np.array(glob(self.testGTLoc))

		# loads the data, SAVED AS NP.ARRAYS
		self.trainSpecs = np.stack([np.load(file) for file in self.trainSpecsFiles])
		self.trainGT = np.stack([np.load(file) for file in self.trainGTFiles])
		self.testSpecs = np.stack([np.load(file) for file in self.testSpecsFiles])
		self.testGT = np.stack([np.load(file) for file in self.testGTFiles])		


	'''
	Name = getBatch
	Purpose = to grab training data with the corresponding ground truth labels
	Parameters = itself, size = how much data desired, rand = randomly choose or not
	Returns = an array of [batchSpecs = data np.array, batchGT = ground truth np.array
	'''
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
		
		# fill arrays to be returned
		batchSpecs = self.trainSpecs[[i for i in index]]
		batchGT = self.trainGT[[i for i in index]]
		
		return [batchSpecs, batchGT]

	'''
	Name = getTest
	Purpose = to grab test data with the corresponding ground truth labels
	Parameters = itself, size = how much data desired, all = all test data or random
	Returns = an array of [test_specs = data np.array, test_gt = ground truth np.array
	'''
	def getTest(self, size, all=False):

		test_specs = None
		test_gt = None
		
		if(all): # use all data, so just return all data
			test_specs = self.testSpecs
			test_gt = self.testGT
		else:	# randomly choose ***not completed yet!!*** only works for all
			index = np.random.randint(self.trainSpecs.shape[0], size=size)

		return [test_specs, test_gt]
		

