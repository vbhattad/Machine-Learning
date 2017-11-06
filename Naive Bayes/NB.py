import math
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
	## Inputs ## 
	# x - 1D numpy ndarray
	
	## Outputs ##
	# log_product - float
	log_product = 0
	for i in x:
		# The product of logs is sum of logs. Taking log to the base 10.
		log_product += i
	return log_product

# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters beta_0 and beta_1, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, beta_0, beta_1):
	## Inputs ## 
	# XTrain - (n by V) numpy ndarray
	# yTrain - 1D numpy ndarray of length n
	# alpha - float
	# beta - float
	
	
	## Outputs ##
	# D - (2 by V) numpy ndarray
	D = np.zeros(shape=(2,XTrain.shape[1]))
	yzero = (yTrain == 0)
	yone = (yTrain == 1)
	for i in range(XTrain.shape[1]):
		alpha0 = np.sum(((XTrain[:,i] == 1) & yzero))
		alpha1 = np.sum(yzero) - alpha0
		theta0 = (alpha0+beta_0-1)/float(alpha0+alpha1+beta_0+beta_1-2)

		alpha0 = np.sum(((XTrain[:,i] == 1) & yone))
		alpha1 = np.sum(yone) - alpha0
		theta1 = (alpha0+beta_0-1)/float(alpha0+alpha1+beta_0+beta_1-2)
		D[0][i] = theta0
		D[1][i] = theta1
	# print D[0][0]
	return D


	
# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
	## Inputs ## 
	# yTrain - 1D numpy ndarray of length n

	## Outputs ##
	# p - float
	# MLE is given as below
	p = np.sum(yTrain) / np.size(yTrain)
	return p

# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
	## Inputs ## 
	# D - (2 by V) numpy ndarray
	# p - float
	# XTest - (m by V) numpy ndarray
	
	## Outputs ##
	# yHat - 1D numpy ndarray of length m
	yHat = []
	yprior = np.log(p) - np.log(1-p)
	theta = np.log(D[0,:]) - np.log(D[1,:])
	oneminustheta = np.log((1-D[0,:])) - np.log((1-D[1,:]))
	for i in range(XTest.shape[0]):
		# print D.shape
		temp = theta * XTest[i,:]
		one = logProd(temp)
		temp = oneminustheta * (1 - XTest[i,:])
		second = logProd(temp)
		prob = yprior + one + second
		prob0 = np.power(10, prob)
		# print prob0
		if prob0 > 1:
			yHat.append(0)
		else:
			yHat.append(1)
	yHat = np.array(yHat)
	return yHat
		

# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
	## Inputs ## 
	# yHat - 1D numpy ndarray of length m
	# yTruth - 1D numpy ndarray of length m
	
	## Outputs ##
	# error - float
	trues = sum(yHat == yTruth)
	error = trues/float(yTruth.shape[0])
	return 1-error
