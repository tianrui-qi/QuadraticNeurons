from Gaussian import Gaussian
from EM import EM
from NN import NN

import os
import numpy as np

D = 2
K = 7
sample_number = 50

def set_sample():
    mu_set = [(np.random.random(D) - 0.5) * 10 for i in range(K - 1)]
    mu_set.insert(0, [0.0, 0.0])
    cov_set = [[[30., 0.0], [0.0, 30.]]]
    for i in range(K - 1):
        a = np.random.random((D, D)) * 2 - 1
        cov = np.dot(a, a.T) + np.dot(a, a.T)
        cov_set.append(cov)
    N_k = [np.random.randint(10000, 15000) for k in range(K - 1)]
    N_k.insert(0, 50000)

    gaussian = Gaussian(N_k, mu_set, cov_set)

    return gaussian.split_sample()

def set_method(j):
    neuron_num_1     = {0: K}
    neuron_num_2_10  = {0: 10, 1: K}
    neuron_num_2_100 = {0: 100, 1: K}

    activation_func_1 = {0: NN.softmax}
    activation_func_2 = {0: NN.relu, 1: NN.softmax}

    if j == 0:
        string = "        EM"
        method = EM(K)
    elif j == 1:
        string = "    Q({}-{})".format(D, K)
        method = NN(D, neuron_num_1, activation_func_1, NN_type="QNN")
    elif j == 2:
        string = "L({}-100-{})".format(D, K)
        method = NN(D, neuron_num_2_100, activation_func_2, NN_type="LNN")
    elif j == 3:
        string = " L({}-10-{})".format(D, K)
        method = NN(D, neuron_num_2_10, activation_func_2, NN_type="LNN")
    else:
        string = "    L({}-{})".format(D, K)
        method = NN(D, neuron_num_1, activation_func_1, NN_type="LNN")

    return string, method


if not os.path.exists('complex'): os.mkdir('complex')

accuracy = np.zeros([sample_number, 5])
precision = np.zeros([sample_number, 5])
recall = np.zeros([sample_number, 5])
train_time = np.zeros([sample_number, 5])

for S in range(sample_number):
    print(D, K, S)
    print("   method  | accuracy  | precision | recall")

    train_point, train_label, valid_point, valid_label, \
    test_point, test_label = set_sample()

    for j in range(5):
        string, method = set_method(j)
        if j == 0: method.train(train_point)
        else: method.train(train_point, train_label, valid_point, valid_label)

        print("%s | %2.6f | %2.6f | %2.6f"
              % (string,
                 method.accuracy(test_point, test_label) * 100,
                 method.precision(test_point, test_label) * 100,
                 method.recall(test_point, test_label) * 100))

    np.savetxt("complex/D={}, K={}, accuracy.csv".format(D, K),
               accuracy, delimiter=",")
    np.savetxt("complex/D={}, K={}, precision.csv".format(D, K),
               precision, delimiter=",")
    np.savetxt("complex/D={}, K={}, recall.csv".format(D, K),
               recall, delimiter=",")
    np.savetxt("complex/D={}, K={}, train_time.csv".format(D, K),
               train_time, delimiter=",")
