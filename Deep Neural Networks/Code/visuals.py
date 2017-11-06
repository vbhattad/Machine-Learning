import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import math
import scipy.io
import copy
import pickle
import matplotlib.pyplot as plt

# # snap = scipy.io.loadmat('lenet.mat')
# objects = []
# with (open("lenet.mat", "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break
# print objects
train_cost = []
test_cost = []
train_acc = []
test_acc = []
# objects = []
# with (open("out-clean.txt", "rb")) as openfile:
#     while True:
#         try:
            
#             l = openfile.readline()
#             if(l == "###"):
#                 break
#             t = l.split(",")
#             k = t[1].split(" ")
#             train_acc.append(float(k[0]))
#             train_cost.append(float(t[0]))

#             l = openfile.readline()
#             t = l.split(",")
#             k = t[1].split(" ")
#             test_acc.append(float(k[0]))
#             test_cost.append(float(t[0]))
#             # objects.append(l)
#         except EOFError:
#             break
X_label = [x for x in range(100,10000,100)]
# print len(X_label)
pickle_file = open('train_acc.pickle', 'rb')
train_acc = pickle.load( pickle_file)
pickle_file.close()

pickle_file = open('test_acc.pickle', 'rb')
test_acc = pickle.load( pickle_file)
pickle_file.close()

pickle_file = open('test_cost.pickle', 'rb')
test_cost = pickle.load(pickle_file)
pickle_file.close()

pickle_file = open('train_cost.pickle', 'rb')
train_cost = pickle.load(pickle_file)
pickle_file.close()
# x, y = zip(*lists) # unpack a list of pairs into two tuples

# plt.plot(X_label, train_acc)
# plt.plot(X_label, test_acc)
# plt.title('Training and Test Accuracy')
# plt.xticks(np.arange(0,10001,1000))
# plt.yticks(np.arange(min(train_acc), max(train_acc), 0.01))
# plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
# plt.savefig('TrainVsTest Accuracy.PNG')
# plt.show()

plt.plot(X_label, train_cost)
plt.plot(X_label, test_cost)
plt.xticks(np.arange(0,10001,1000))
plt.yticks(np.arange(min(train_cost), max(train_cost), 0.05))
plt.title('Training and Test Cost')
plt.legend(['Training Cost', 'Test Cost'], loc='upper right')
plt.savefig('TrainVsTest Cost.PNG')
plt.show()