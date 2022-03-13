from Gaussian import Gaussian
from EM import EM
from LNN import LNN
from QNN import QNN
from Visual import Visual

import os
import numpy as np
import matplotlib.pyplot as plt

D   = 2     # dimension of sample data point
K   = 3     # number of Gaussian / classifications

def set_method(j):
    LNN_neuron_num_1 = {0: K}
    LNN_neuron_num_2_10 = {0: 10, 1: K}
    LNN_neuron_num_2_50 = {0: 50, 1: K}
    LNN_neuron_num_2_100 = {0: 100, 1: K}
    QNN_neuron_num_1 = {0: K}

    LNN_activation_func_1 = {0: LNN.softmax}
    LNN_activation_func_2 = {0: LNN.relu, 1: LNN.softmax}
    QNN_activation_func_1 = {0: QNN.softmax}

    if j == 0:
        string = "EM"
        method = EM(K)
    elif j == 1:
        string = "L({}-{})".format(D, K)
        method = LNN(D, LNN_neuron_num_1, LNN_activation_func_1)
    elif j == 2:
        string = "L({}-10-{})".format(D, K)
        method = LNN(D, LNN_neuron_num_2_10, LNN_activation_func_2)
    elif j == 3:
        string = "L({}-50-{})".format(D, K)
        method = LNN(D, LNN_neuron_num_2_50, LNN_activation_func_2)
    elif j == 4:
        string = "L({}-100-{})".format(D, K)
        method = LNN(D, LNN_neuron_num_2_100, LNN_activation_func_2)
    else:
        string = "Q({}-{})".format(D, K)
        method = QNN(D, QNN_neuron_num_1, QNN_activation_func_1)

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
        # time
        train_time.append(method.train_time)
        test_time.append(method.test_time)

        # loss
        if j != 0:
            train_loss.append(method.train_loss)
            valid_loss.append(method.valid_loss)
            test_loss.append(method.test_loss)

        # accuracy
        train_accuracy.append(method.train_accuracy)
        valid_accuracy.append(method.valid_accuracy)
        test_accuracy.append(method.test_accuracy)


def main():
    if not os.path.exists('special'): os.mkdir('special')
    if not os.path.exists('special/fig'): os.mkdir('special/fig')
    if not os.path.exists('special/result'): os.mkdir('special/result')

    for i in range(20):
        """ generate sample and label """

        N_k = [np.random.randint(20000, 30000) for k in range(K)]
        mu_set = np.array([[0.0, 0.0],
                           [-1.0, 2.0],
                           [1.0, -2.0]])
        cov_set = np.array([[[40., 0.0], [0.0, 40.]],
                            [[1.0, 0.5], [0.5, 0.5]],
                            [[0.5, 0.5], [0.5, 1.0]]])
        gaussian = Gaussian(N_k, mu_set, cov_set)
        point, label, = gaussian.generate_sample()

        """ initialize object visualization """

        visual = Visual(point, label, mu_set, cov_set)
        visual.plot_sample().savefig("special/fig/{}_sample".format(i))

        """ split sample into three part: train, validation, and test set """

        index_1 = int(0.6 * len(point))
        index_2 = int(0.8 * len(point))
        train_point = np.array([point[i] for i in range(index_1)])
        train_label = np.array([label[i] for i in range(index_1)])
        valid_point = np.array([point[i] for i in range(index_1, index_2)])
        valid_label = np.array([label[i] for i in range(index_1, index_2)])
        test_point = np.array([point[i] for i in range(index_2, len(point))])
        test_label = np.array([label[i] for i in range(index_2, len(point))])

        """ variable use to store result including time and accuracy """

        train_time, test_time = [], []
        train_loss, valid_loss, test_loss = [], [], []
        train_accuracy, valid_accuracy, test_accuracy = [], [], []

        """ train """

        for j in range(6):
            string, method = set_method(j)
            if j == 0:
                accuracy = 0
                while accuracy < 0.8:
                    method.train(train_point, train_label=train_label,
                                 valid_point=valid_point, valid_label=valid_label,
                                 test_point=test_point, test_label=test_label, )
                    accuracy = method.test_accuracy[0]
            else:
                optimizer_para = {
                    "lr": 0.01,  # float, for all optimizer
                    "decay_rate": 0.99,  # float, for optimizer "RMSprop"
                    "beta1": 0.9,  # float, for optimizer "Adam"
                    "beta2": 0.999,  # float, for optimizer "Adam"
                    "iter": 0
                }
                method.train(train_point, train_label, optimizer_para,
                             valid_point=valid_point, valid_label=valid_label,
                             test_point=test_point, test_label=test_label)
            fig = visual.plot_DB(method)
            fig.savefig("special/fig/{}_{}".format(i, string))

            save(train_time, test_time,
                 train_loss, valid_loss, test_loss,
                 train_accuracy, valid_accuracy, test_accuracy,
                 method=method, j=j)

        save(train_time, test_time,
             train_loss, valid_loss, test_loss,
             train_accuracy, valid_accuracy, test_accuracy, i=i)

    for i in range(20):
        result = np.loadtxt("special/result/{}_test_accuracy.csv".format(i),
                            delimiter=",")
        if result[1] < result[2] < result[3] < result[4] < result[5] < result[0]:
            print(i)


if __name__ == "__main__":
    main()
