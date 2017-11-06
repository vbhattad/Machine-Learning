#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 50):
        '''
        Initializes Parameters of the  Logistic Regression Model
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
  
    

    
    
    def calculateGradient(self, weight, X, Y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
        
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is (d+1)-by-1 dimensional numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an (d+1)-by-1 dimensional numpy matrix
        '''
        Gradient = np.zeros((X.shape[1],1))
        Gradient = np.dot(X.T, (self.sigmoid(np.dot(X,weight))-Y)) + regLambda*weight
        Gradient[0] -= regLambda*weight[0]
        return Gradient.reshape((X.shape[1],1))

    def sigmoid(self, Z):
        '''
        Computes the Sigmoid Function  
        Arguments:
            A n-by-1 dimensional numpy matrix
        Returns:
            A n-by-1 dimensional numpy matrix
       
        '''
        # print Z
        denom = np.exp(-Z)
        sigmoid = 1/(1+denom)
        sigmoid = np.array(sigmoid)
        return sigmoid

    def update_weight(self,X,Y,weight):
        '''
        Updates the weight vector.
        Arguments:
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is a d+1-by-1 dimensional numpy matrix
        Returns:
            updated weight vector : (d+1)-by-1 dimensional numpy matrix
        '''
        
        gradient = self.calculateGradient(weight,X,Y,self.regLambda)
        second = self.alpha * gradient
        new_weight = weight - second
        return new_weight
    
    def check_conv(self,weight,new_weight,epsilon):
        '''
        Convergence Based on Tolerance Values
        Arguments:
            weight is a (d+1)-by-1 dimensional numpy matrix
            new_weights is a (d+1)-by-1 dimensional numpy matrix
            epsilon is the Tolerance value we check against
        Return : 
            True if the weights have converged, otherwise False

        '''
        # print np.sqrt(np.sum(np.power((new_weight - weight),2)))
        return np.sqrt(np.sum(np.power((new_weight - weight),2))) <= epsilon
        
    def train(self,X,Y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            Y is an n-by-1 dimensional numpy matrix
        Return:
            Updated Weights Vector: (d+1)-by-1 dimensional numpy matrix
        '''
        # Read Data
        n,d = X.shape
        
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        self.weight = self.new_weight = np.zeros((d+1,1))
        self.new_weight = self.update_weight(X, Y, self.weight)
        for i in range(1,self.maxNumIters):
            # print i
            if self.check_conv(self.weight, self.new_weight, self.epsilon):
                break
            else:
                self.weight = self.new_weight
                self.new_weight = self.update_weight(X, Y, self.weight)
        self.weight = self.new_weight
            # print self.new_weight.shape
        print self.weight.reshape((X.shape[1],1)).shape
        return self.weight.reshape((X.shape[1],1))        

    def predict_label(self, X,weight):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
            weight is a d+1-by-1 dimensional matrix
        Returns:
            an n-by-1 dimensional matrix of the predictions 0 or 1
        '''
        #data
        n=X.shape[0]
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        res = self.sigmoid(np.dot(X, weight))
        # print res
        # z = np.sum(weight.T*X, axis=1)
        # res = self.sigmoid(z)
        result = (res >= 0.5)        
        result = np.array(result.astype(int))
        return result
    
    def calculateAccuracy (self, Y_predict, Y_test):
        '''
        Computes the Accuracy of the model
        Arguments:
            Y_predict is a n-by-1 dimensional matrix (Predicted Labels)
            Y_test is a n-by-1 dimensional matrix (True Labels )
        Returns:
            Scalar value for accuracy in the range of 0 - 100 %
        '''
        # print (Y_predict)
        num = np.sum(Y_predict == Y_test)
        den = float(len(Y_test))
        # print num, den
        Accuracy = num/den
        
        return Accuracy*100
    
        