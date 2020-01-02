import numpy as np
import math
import hickle as hkl

def sigmoid(x, deriv=False):
	if (deriv==True):
		return sigmoid(x)*(1-sigmoid(x))
	return 1/(1+np.exp(-x))

class Network:
	def __init__(self, inputSet, needResult):
		self.input = inputSet
		self.firstWeights = np.random.random((6, 20))
		self.secondWeights = np.random.random((20, 10))
		self.outputLayer = needResult
	
	def setInput(self, inputSet):
		self.input = inputSet
	
	def feedForward(self):
		self.hiddenLayer = sigmoid(np.dot(self.input, self.firstWeights))
		self.output = sigmoid(np.dot(self.hiddenLayer,self.secondWeights))

	def backLoss(self):
		secondError = self.outputLayer - self.output
		d_secondWeights = secondError * sigmoid(self.output,True)
		d_firstWeights = d_secondWeights.dot(self.secondWeights.T) * sigmoid(self.hiddenLayer,True)
		
		self.secondWeights += self.hiddenLayer.T.dot(d_secondWeights)
		self.firstWeights += self.input.T.dot(d_firstWeights)
		
	def saveWeights(self):
		hkl.dump(self.firstWeights, "data/firstLayer.hkl")
		hkl.dump(self.secondWeights, "data/secondLayer.hkl")
	
	def loadWeights(self):
		self.firstWeights = hkl.load("data/firstLayer.hkl")
		self.secondWeights = hkl.load("data/secondLayer.hkl")

if __name__ == ("__main__"):
	n = 10
	template = '{:.' + str(n) + 'f}'	
	testNumber = [0,0.5,0.5,1,0,0.6]
	
	trainSet = np.array([[0.5, 1, 1, 0, 0.5, 0.2],
						 [1, 1, 0.5, 1, 1, 0.4],
						 [1, 0, 0, 0, 1, 0.7]
						])

	result = np.array([[1,0,0.5]]).T
	
	
	myNetwork = Network(trainSet, result)
	myNetwork.loadWeights()
	myNetwork.feedForward()
	myNetwork.saveWeights()
	
	print("Learning outputs:")
	print(template.format(myNetwork.output[0][0]))
	print(template.format(myNetwork.output[1][0]))
	print(template.format(myNetwork.output[2][0]))
	
	print("--------------------------------------")
	
	testSet = np.array([testNumber])
	myNetwork.setInput(testSet)
	myNetwork.feedForward()
	
	print("Test outputs: " + template.format(myNetwork.output[0][0]))
	
	
