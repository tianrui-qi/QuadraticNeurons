from Gaussian import Gaussian
from EM import EM
from LNN import LNN
from QNN import QNN

import os
import numpy as np

sample_number = 50


def set_method(j):
    LNN_neuron_num_1     = {0: K}
    LNN_neuron_num_2_10  = {0: 10, 1: K}
    LNN_neuron_num_2_100 = {0: 100, 1: K}
    QNN_neuron_num_1     = {0: K}

    LNN_activation_func_1 = {0: LNN.softmax}
    LNN_activation_func_2 = {0: LNN.relu, 1: LNN.softmax}
    QNN_activation_func_1 = {0: QNN.softmax}

    if j == 0:
        string = "        EM"
        method = EM(K)
    elif j == 1:
        string = "    Q({}-{})".format(D, K)
        method = QNN(D, QNN_neuron_num_1, QNN_activation_func_1)
    elif j == 2:
        string = "L({}-100-{})".format(D, K)
        method = LNN(D, LNN_neuron_num_2_100, LNN_activation_func_2)
    elif j == 3:
        string = " L({}-10-{})".format(D, K)
        method = LNN(D, LNN_neuron_num_2_10, LNN_activation_func_2)
    else:
        string = "    L({}-{})".format(D, K)
        method = LNN(D, LNN_neuron_num_1, LNN_activation_func_1)

    return string, method


if not os.path.exists('complex'): os.mkdir('complex')
for D in (2, 3):
    for K in (5, 8):
        test_accuracy = np.zeros([sample_number, 5])
        test_time     = np.zeros([sample_number, 5])
        train_time    = np.zeros([sample_number, 5])

        for S in range(sample_number):
            print(D, K, S)

            """ Set mu, cov """

            mu_set = np.array([(np.random.random(D) - 0.5) * 10
                               for i in range(K)])
            cov_set = []
            for i in range(K):
                a = np.random.random((D, D)) * 2 - 1
                cov = np.dot(a, a.T) + np.dot(a, a.T)
                cov_set.append(cov)

            """ Generate Sample """

            N_k = [np.random.randint(15000, 20000) for k in range(K)]
            gaussian = Gaussian(N_k, mu_set, cov_set)
            point, label, = gaussian.generate_sample()

            index_1 = int(0.6 * len(point))
            index_2 = int(0.8 * len(point))
            train_point = np.array([point[i] for i in range(index_1)])
            train_label = np.array([label[i] for i in range(index_1)])
            valid_point = np.array([point[i] for i in range(index_1, index_2)])
            valid_label = np.array([label[i] for i in range(index_1, index_2)])
            test_point = np.array([point[i] for i in range(index_2, len(point))])
            test_label = np.array([label[i] for i in range(index_2, len(point))])

            """ train """

            for j in range(5):
                string, method = set_method(j)
                if j == 0:
                    method.train(test_point, train_label=test_label,
                                 valid_point=test_point, valid_label=test_label,
                                 test_point=test_point, test_label=test_label, )
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
                test_accuracy[S][j] = method.test_accuracy[0]
                test_time[S][j]     = method.test_time[0]
                train_time[S][j]    = np.max(method.train_time)

                print("{}\t{}".format(string, test_accuracy[S][j] * 100))

            np.savetxt("complex/D={}, K={}, test_accuracy.csv".format(D, K),
                       test_accuracy, delimiter=",")
            np.savetxt("complex/D={}, K={}, test_time.csv".format(D, K),
                       test_time, delimiter=",")
            np.savetxt("complex/D={}, K={}, train_time.csv".format(D, K),
                       train_time, delimiter=",")
