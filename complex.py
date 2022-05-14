from Visual import Visual
from Gaussian import Gaussian
from EM import EM
from NN import NN

import os
import numpy as np

D = 3
K = 5
sample_number = 50


def set_sample():
    mu_set = [(np.random.random(D) - 0.5) * 10 for i in range(K - 1)]
    # mu_set.insert(0, [0.0, 0.0])
    # cov_set = [[[40., 0.0], [0.0, 40.]]]
    mu_set.insert(0, [0.0, 0.0, 0.0])
    cov_set = [[[40., 0.0, 0.0], [0.0, 40., 0.0], [0.0, 0.0, 40.]]]
    for i in range(K - 1):
        a = np.random.random((D, D)) * 2 - 1
        cov = np.dot(a, a.T) + np.dot(a, a.T)
        cov_set.append(cov)
    N_k = [np.random.randint(3000, 6000) for k in range(K - 1)]
    N_k.insert(0, 20000)

    gaussian = Gaussian(N_k, mu_set, cov_set)
    visual = Visual(gaussian.point, gaussian.label, mu_set, cov_set)

    return gaussian, visual


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


def main(accuracy, precision, recall, train_time):

    for S in range(sample_number):
        print(D, K, S, "\n   method  | accuracy  | precision | recall    | time")

        gaussian, visual = set_sample()
        train_point, train_label, valid_point, valid_label, \
        test_point, test_label = gaussian.split_sample()
        visual.plot_sample().savefig("complex/fig/{}_sample".format(S))

        for j in range(5):
            string, method = set_method(j)
            if j == 0:
                method.train(train_point)
                method.order_correction(train_point, train_label)
            elif j == 1:
                method.train(train_point, train_label, valid_point, valid_label)
            else:
                method.train(train_point, train_label, valid_point, valid_label,
                             stop_point=200)

            accuracy[S][j] = method.accuracy(test_point, test_label)
            precision[S][j] = method.precision(test_point, test_label)
            recall[S][j] = method.recall(test_point, test_label)
            if j != 0:
                train_time[S][j] = max(method.train_time)

            print("%s | %2.6f | %2.6f | %2.6f | %2.6f"
                  % (string, accuracy[S][j] * 100, precision[S][j] * 100,
                     recall[S][j] * 100, train_time[S][j]))

            if D == 2:
                visual.plot_DB(method)\
                    .savefig("complex/fig/{}_{}_DB".format(S, string))


if __name__ == "__main__":
    if not os.path.exists('complex'): os.mkdir('complex')
    if not os.path.exists('complex/fig'): os.mkdir('complex/fig')
    if not os.path.exists('complex/result'): os.mkdir('complex/result')

    accuracy   = np.zeros([sample_number, 5])
    precision  = np.zeros([sample_number, 5])
    recall     = np.zeros([sample_number, 5])
    train_time = np.zeros([sample_number, 5])

    main(accuracy, precision, recall, train_time)

    np.savetxt("complex/result/accuracy.csv".format(D, K),
               accuracy, delimiter=",")
    np.savetxt("complex/result/precision.csv".format(D, K),
               precision, delimiter=",")
    np.savetxt("complex/result/recall.csv".format(D, K),
               recall, delimiter=",")
    np.savetxt("complex/result/train_time.csv".format(D, K),
               train_time, delimiter=",")
