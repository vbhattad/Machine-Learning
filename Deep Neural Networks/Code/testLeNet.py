#######################################################################################
#######       DO NOT MODIFY, DEFINITELY READ THROUGH ALL OF THE CODE            #######
#######################################################################################
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cnn_lenet
import pickle
import copy
import random
import matplotlib.pyplot as plt
import time
def get_lenet():
  """Define LeNet

  Explanation of parameters:
  type: layer type, supports convolution, pooling, relu
  channel: input channel
  num: output channel
  k: convolution kernel width (== height)
  group: split input channel into several groups, not used in this assignment
  """

  layers = {}
  layers[1] = {}
  layers[1]['type'] = 'DATA'
  layers[1]['height'] = 28
  layers[1]['width'] = 28
  layers[1]['channel'] = 1
  layers[1]['batch_size'] = 64

  layers[2] = {}
  layers[2]['type'] = 'CONV'
  layers[2]['num'] = 20
  layers[2]['k'] = 5
  layers[2]['stride'] = 1
  layers[2]['pad'] = 0
  layers[2]['group'] = 1

  layers[3] = {}
  layers[3]['type'] = 'POOLING'
  layers[3]['k'] = 2
  layers[3]['stride'] = 2
  layers[3]['pad'] = 0

  layers[4] = {}
  layers[4]['type'] = 'CONV'
  layers[4]['num'] = 50
  layers[4]['k'] = 5
  layers[4]['stride'] = 1
  layers[4]['pad'] = 0
  layers[4]['group'] = 1

  layers[5] = {}
  layers[5]['type'] = 'POOLING'
  layers[5]['k'] = 2
  layers[5]['stride'] = 2
  layers[5]['pad'] = 0

  layers[6] = {}
  layers[6]['type'] = 'IP'
  layers[6]['num'] = 500
  layers[6]['init_type'] = 'uniform'

  layers[7] = {}
  layers[7]['type'] = 'RELU'

  layers[8] = {}
  layers[8]['type'] = 'LOSS'
  layers[8]['num'] = 10
  return layers


def main():
  # define lenet
  layers = get_lenet()

  # load data
  # change the following value to true to load the entire dataset
  fullset = True
  print("Loading MNIST Dataset...")
  xtrain, ytrain, xval, yval, xtest, ytest = cnn_lenet.load_mnist(fullset)
  print("MNIST Dataset Loading Complete!\n")

  xtrain = np.hstack([xtrain, xval])
  ytrain = np.hstack([ytrain, yval])
  m_train = xtrain.shape[1]

  # cnn parameters
  batch_size = 64
  mu = 0.9
  epsilon = 0.01
  gamma = 0.0001
  power = 0.75
  weight_decay = 0.0005
  w_lr = 1
  b_lr = 2

  test_interval = 15
  display_interval = 15
  snapshot = 5000
  max_iter = 30

  # initialize parameters
  print("Initializing Parameters...")
  params = cnn_lenet.init_convnet(layers)
  param_winc = copy.deepcopy(params)
  print("Initilization Complete!\n")

  for l_idx in range(1, len(layers)):
    param_winc[l_idx]['w'] = np.zeros(param_winc[l_idx]['w'].shape)
    param_winc[l_idx]['b'] = np.zeros(param_winc[l_idx]['b'].shape)

  # learning iterations
  random.seed(100000)
  indices = range(m_train)
  random.shuffle(indices)

  print("Training Started. Printing report on training data every " + str(display_interval) + " steps.")
  print("Printing report on test data every " + str(test_interval) + " steps.\n")
  train_acc_100 = []
  test_acc_100 = []
  train_cost_100 = []
  test_cost_100 = []

  program_starts = time.time()
  for step in range(max_iter):
    # get mini-batch and setup the cnn with the mini-batch
    start_idx = step * batch_size % m_train
    end_idx = (step+1) * batch_size % m_train
    if start_idx > end_idx:
      random.shuffle(indices)
      continue
    idx = indices[start_idx: end_idx]
  
    layers[2]['iteration'] = step
    [cp, param_grad] = cnn_lenet.conv_net(params,
                                          layers,
                                          xtrain[:, idx],
                                          ytrain[idx], True)

    # we have different epsilons for w and b
    w_rate = cnn_lenet.get_lr(step, epsilon*w_lr, gamma, power)
    b_rate = cnn_lenet.get_lr(step, epsilon*b_lr, gamma, power)
    params, param_winc = cnn_lenet.sgd_momentum(w_rate,
                           b_rate,
                           mu,
                           weight_decay,
                           params,
                           param_winc,
                           param_grad)

    # display training loss
    if (step+1) % display_interval == 0:
      train_cost_100.append(cp['cost'])
      train_acc_100.append(cp['percent'])
      print 'training_cost = %f training_accuracy = %f' % (cp['cost'], cp['percent']) + ' current_step = ' + str(step + 1)
      now = time.time()
      print "It has been {0} seconds since the training started".format(now - program_starts) 
    
    # display test accuracy
    if (step+1) % test_interval == 0:
      layers[1]['batch_size'] = xtest.shape[1]
      cptest, _ = cnn_lenet.conv_net(params, layers, xtest, ytest, False)
      layers[1]['batch_size'] = 64
      test_cost_100.append(cptest['cost'])
      test_acc_100.append(cptest['percent'])
      print 'test_cost = %f test_accuracy = %f' % (cptest['cost'], cptest['percent']) + ' current_step = ' + str(step + 1) + '\n'

    # save params peridocally to recover from any crashes
    if (step+1) % snapshot == 0:
      pickle_path = 'lenet.mat'
      pickle_file = open(pickle_path, 'wb')
      pickle.dump(params, pickle_file)
      pickle_file.close()

  # pickle_file = open('train_acc.pickle', 'wb')
  # pickle.dump(train_acc_100, pickle_file)
  # pickle_file.close()

  # pickle_file = open('test_acc.pickle', 'wb')
  # pickle.dump(test_acc_100, pickle_file)
  # pickle_file.close()

  # pickle_file = open('test_cost.pickle', 'wb')
  # pickle.dump(test_cost_100, pickle_file)
  # pickle_file.close()

  # pickle_file = open('train_cost.pickle', 'wb')
  # pickle.dump(train_cost_100, pickle_file)
  # pickle_file.close()

  # lists = sorted(train_cost_100.items()) # sorted by key, return a list of tuples
  # x, y = zip(*lists) # unpack a list of pairs into two tuples
  # plt.plot(x, y)

  # lists = sorted(test_cost_100.items()) # sorted by key, return a list of tuples
  # a, b = zip(*lists) # unpack a list of pairs into two tuples
  # plt.plot(a, b)

  # plt.legend(['Training cost', 'Test Cost'], loc='upper left')
  # plt.savefig('TrainVsTest Cost.PNG')
  # plt.show()

  # lists = sorted(train_acc_100.items()) # sorted by key, return a list of tuples
  # x, y = zip(*lists) # unpack a list of pairs into two tuples
  # plt.plot(x, y)

  # lists = sorted(test_acc_100.items()) # sorted by key, return a list of tuples
  # a, b = zip(*lists) # unpack a list of pairs into two tuples
  # plt.plot(a, b)

  # plt.legend(['Training Accuracy', 'Test Accuracy'], loc='upper left')
  # plt.savefig('TrainVsTest Accuracy.PNG')
  # plt.show()

if __name__ == '__main__':
  main()
