from Gaussian import Gaussian
from EM import EM
from NN import NN
from Visual import Visual

import os
import numpy as np
import matplotlib.pyplot as plt

D   = 2     # dimension of sample data point
K   = 3     # number of Gaussian / classifications

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

def save(train_time, test_time,
         train_loss, valid_loss, test_loss,
         train_accuracy, valid_accuracy, test_accuracy,
         method=None, i=None, j=None):
    if j is None and method is None:
        # time
        np.savetxt("special/result/{}_train_time.csv".format(i),
                   np.array(train_time).T, delimiter=",")
        np.savetxt("special/result/{}_test_time.csv".format(i),
                   np.array(test_time).T, delimiter=",")

        # loss
        np.savetxt("special/result/{}_train_loss.csv".format(i),
                   np.array(train_loss).T, delimiter=",")
        np.savetxt("special/result/{}_valid_loss.csv".format(i),
                   np.array(valid_loss).T, delimiter=",")
        np.savetxt("special/result/{}_test_loss.csv".format(i),
                   np.array(test_loss).T, delimiter=",")

        # accuracy
        np.savetxt("special/result/{}_train_accuracy.csv".format(i),
                   np.array(train_accuracy).T, delimiter=",")
        np.savetxt("special/result/{}_valid_accuracy.csv".format(i),
                   np.array(valid_accuracy).T, delimiter=",")
        np.savetxt("special/result/{}_test_accuracy.csv".format(i),
                   np.array(test_accuracy).T, delimiter=",")
    if i is None:
        if j == 0: return
        # time
        train_time.append(method.train_time)

        # loss
        train_loss.append(method.train_loss)
        valid_loss.append(method.valid_loss)

        # accuracy
        train_accuracy.append(method.train_accuracy)
        valid_accuracy.append(method.valid_accuracy)

def main():
    N_k = [np.random.randint(2000, 3000) for k in range(K-1)]
    N_k.insert(0, 10000)
    mu_set = np.array([[0.0, 0.0],
                       [-1.0, 2.0],
                       [1.0, -2.0]])
    cov_set = np.array([[[30., 0.0], [0.0, 30.]],
                        [[1.0, 0.5], [0.5, 0.5]],
                        [[0.5, 0.5], [0.5, 1.0]]])

    gaussian = Gaussian(N_k, mu_set, cov_set)
    train_point, train_label, valid_point, valid_label, \
    test_point, test_label = gaussian.split_sample()

    """ initialize object visualization """

    visual = Visual(gaussian.point, gaussian.label, mu_set, cov_set)
    visual.plot_sample().savefig("special/fig/sample")

    """ variable use to store result including time and accuracy """

    train_time, test_time = [], []
    train_loss, valid_loss, test_loss = [], [], []
    train_accuracy, valid_accuracy, test_accuracy = [], [], []

    """ train """

    print("   method  | accuracy  | precision | recall")
    for j in range(5):
        string, method = set_method(j)
        if j == 0: method.train(train_point)
        else: method.train(train_point, train_label, valid_point, valid_label)

        print("%s | %2.6f | %2.6f | %2.6f"
              % (string,
                 method.accuracy(test_point, test_label) * 100,
                 method.precision(test_point, test_label) * 100,
                 method.recall(test_point, test_label) * 100))

        visual.plot_DB(method).savefig("special/fig/{}_DB".format(string))

        save(train_time, test_time,
             train_loss, valid_loss, test_loss,
             train_accuracy, valid_accuracy, test_accuracy,
             method=method, j=j)


if __name__ == "__main__":
    if not os.path.exists('special'): os.mkdir('special')
    if not os.path.exists('special/fig'): os.mkdir('special/fig')
    if not os.path.exists('special/result'): os.mkdir('special/result')

    main()
