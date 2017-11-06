
# import pandas as pd
import numpy as np
from LogReg import LogisticRegression


#Read the data

X_Train = np.genfromtxt('../data/XTrain.csv', delimiter=",")
Y_Train = np.genfromtxt('../data/YTrain.csv',delimiter=",")
Y_Train = Y_Train.reshape((Y_Train.shape[0],1))
X_Test = np.genfromtxt('../data/XTest.csv', delimiter=",")
Y_Test = np.genfromtxt('../data/YTest.csv', delimiter=",")
Y_Test = Y_Test.reshape((Y_Test.shape[0],1))

LogReg=LogisticRegression()

#Uncomment this only after your Logistic Regression Class has been Completed
# X_Train = np.array([[1,2],[1,3],[2,3],[2,2],[3,3],[4,4]])
# Y_Train = np.array([0,0,0,1,1,1])
weight=LogReg.train(X_Train,Y_Train)
print weight
Y_predict=np.array(LogReg.predict_label(X_Test,weight))
print (LogReg.calculateAccuracy(Y_predict,Y_Test))
