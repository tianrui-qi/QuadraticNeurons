from Gaussian import Gaussian
from LNN import LNN
from QNN import QNN
from Visual import Visual

import os
import numpy as np


D   = 2     # dimension of sample data point
K   = 3     # number of Gaussian / classifications
N_k = [np.random.randint(1000, 3000) for k in range(K-1)]

mu_set  = np.array([[-3.0, 1.0],              [-1.0, -3.0]])
cov_set = np.array([[[1.0, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 1.0]]])

run_number = 20
NN_train_number = 10000

LNN_neuron_num_1     = { 0: K }
LNN_neuron_num_2_10  = { 0: 10, 1: K }
LNN_neuron_num_2_50  = { 0: 50, 1: K }
LNN_neuron_num_2_100 = { 0: 100, 1: K }
LNN_neuron_num_2_500 = { 0: 500, 1: K }
QNN_neuron_num_1     = { 0: K }

LNN_activation_func_1 = { 0: LNN.softmax }
LNN_activation_func_2 = { 0: LNN.relu, 1: LNN.softmax }
QNN_activation_func_1 = { 0: QNN.softmax }
QNN_activation_func_2 = { 0: QNN.relu, 1: QNN.softmax }

optimizer_para = {
    "lr":         0.01,     # float, for all optimizer
    "decay_rate": 0.99,     # float, for optimizer "RMSprop"
    "beta1":      0.9,      # float, for optimizer "Adam"
    "beta2":      0.999,    # float, for optimizer "Adam"
    "iter":       0
}


def main():
    if not os.path.exists('special'): os.mkdir('special')
    if not os.path.exists('special/fig'): os.mkdir('special/fig')
    if not os.path.exists('special/NN'): os.mkdir('special/NN')
    if not os.path.exists('special/result'): os.mkdir('special/result')
    if not os.path.exists('special/sample'): os.mkdir('special/sample')

    for i in range(run_number):
        gaussian = Gaussian(mu_set, cov_set)
        train_point, train_label, test_point, test_label, \
        sample_point, sample_label, = gaussian.generate_sample(N_k, bg=True)
        save_sample(gaussian, i)

        visual = Visual(sample_point, sample_label, mu_set, cov_set, bg=True)
        visual.plot_sample().savefig("special/fig/{}_sample".format(i))


        string = "L({}-{})".format(D, K)
        lnn = LNN(D, LNN_neuron_num_1, LNN_activation_func_1)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(lnn, i, string)
        save_result(lnn, i, string)
        visual.plot_LNN_DB(lnn).savefig("special/fig/{}_{}".format(i, string))


        string = "L({}-10-{})".format(D, K)
        lnn = LNN(D, LNN_neuron_num_2_10, LNN_activation_func_2)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(lnn, i, string)
        save_result(lnn, i, string)
        visual.plot_LNN_DB(lnn).savefig("special/fig/{}_{}".format(i, string))


        string = "L({}-50-{})".format(D, K)
        lnn = LNN(D, LNN_neuron_num_2_50, LNN_activation_func_2)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(lnn, i, string)
        save_result(lnn, i, string)
        visual.plot_LNN_DB(lnn).savefig("special/fig/{}_{}".format(i, string))


        string = "L({}-100-{})".format(D, K)
        lnn = LNN(D, LNN_neuron_num_2_100, LNN_activation_func_2)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(lnn, i, string)
        save_result(lnn, i, string)
        visual.plot_LNN_DB(lnn).savefig("special/fig/{}_{}".format(i, string))


        string = "L({}-500-{})".format(D, K)
        lnn = LNN(D, LNN_neuron_num_2_500, LNN_activation_func_2)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(lnn, i, string)
        save_result(lnn, i, string)
        visual.plot_LNN_DB(lnn).savefig("special/fig/{}_{}".format(i, string))


        string = "Q({}-{})".format(D, K)
        qnn = QNN(D, QNN_neuron_num_1, QNN_activation_func_1)
        qnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(qnn, i, string)
        save_result(qnn, i, string)
        visual.plot_QNN_DB(qnn).savefig("special/fig/{}_{}".format(i, string))


def save_sample(gaussian, i):
    file = open("special/sample/{}_parameter.txt".format(i), "w")
    file.write("K = {}, D = {}, N = {}\n"
               .format(gaussian.K, len(gaussian.mu_set[0]), sum(gaussian.N_k)))
    file.write("prio_p = {}\n".format(gaussian.N_k / sum(gaussian.N_k)))
    file.write("N_k = {}\n\n".format(gaussian.N_k))
    for j in range(gaussian.K):
        file.write("Gaussian {}\nmu:\n  {}\ncov:\n  {}\n\n"
                   .format(j, gaussian.mu_set[j], gaussian.cov_set[j]))
    file.close()

    np.savetxt("special/sample/{}_sample_point.csv".format(i),
               gaussian.sample_point, delimiter=",")
    np.savetxt("special/sample/{}_sample_label.csv".format(i),
               gaussian.sample_label, delimiter=",")
    np.savetxt("special/sample/{}_train_point.csv".format(i),
               gaussian.train_point, delimiter=",")
    np.savetxt("special/sample/{}_train_label.csv".format(i),
               gaussian.train_label, delimiter=",")
    np.savetxt("special/sample/{}_test_point.csv".format(i),
               gaussian.test_point, delimiter=",")
    np.savetxt("special/sample/{}_test_label.csv".format(i),
               gaussian.test_label, delimiter=",")


def load_sample(i):
    sample_point = np.loadtxt("special/sample/{}_sample_point.csv".format(i),
                              delimiter=",")
    sample_label = np.loadtxt("special/sample/{}_sample_label.csv".format(i),
                              delimiter=",")
    train_point = np.loadtxt("special/sample/{}_train_point.csv".format(i),
                             delimiter=",")
    train_label = np.loadtxt("special/sample/{}_train_label.csv".format(i),
                             delimiter=",")
    test_point = np.loadtxt("special/sample/{}_test_point.csv".format(i),
                            delimiter=",")
    test_label = np.loadtxt("special/sample/{}_test_label.csv".format(i),
                            delimiter=",")

    return train_point, train_label, test_point, test_label, \
           sample_point, sample_label


def save_result(NN, i, string):
    if not os.path.exists('special'): os.mkdir('special')
    if not os.path.exists('special/result'): os.mkdir('special/result')

    np.savetxt("special/result/{}_{}_train_loss.csv".format(i, string),
               NN.train_loss, delimiter=",")
    np.savetxt("special/result/{}_{}_test_loss.csv".format(i, string),
               NN.test_loss, delimiter=",")
    np.savetxt("special/result/{}_{}_train_accuracy.csv".format(i, string),
               NN.train_accuracy, delimiter=",")
    np.savetxt("special/result/{}_{}_test_accuracy.csv".format(i, string),
               NN.test_accuracy, delimiter=",")


def save_NN(NN, i, string):
    if not os.path.exists('special'): os.mkdir('special')
    if not os.path.exists('special/NN'): os.mkdir('special/NN')

    for key in NN.para.keys():
        np.savetxt("special/NN/{}_{}_para_{}.csv".format(i, string, key),
                   NN.para[key], delimiter=",")
    for key in NN.h.keys():
        np.savetxt("special/NN/{}_{}_h_{}.csv".format(i, string, key),
                   NN.h[key], delimiter=",")
    for key in NN.m.keys():
        np.savetxt("special/NN/{}_{}_m_{}.csv".format(i, string, key),
                   NN.m[key], delimiter=",")
    for key in NN.v.keys():
        np.savetxt("special/NN/{}_{}_v_{}.csv".format(i, string, key),
                   NN.v[key], delimiter=",")


def load_NN(NN, i, string):
    if not os.path.exists('special'): return
    if not os.path.exists('special/NN'): return

    para, h, m, v = {}, {}, {}, {}

    for l in range(NN.L):
        for j in ('w', 'b'):
            if string[0] == 'L':
                key = j + str(l)
                para[key] = np.loadtxt("special/NN/{}_{}_para_{}.csv"
                                       .format(i, string, key), delimiter=",")
                h[key] = np.loadtxt("special/NN/{}_{}_h_{}.csv"
                                    .format(i, string, key), delimiter=",")
                m[key] = np.loadtxt("special/NN/{}_{}_m_{}.csv"
                                    .format(i, string, key), delimiter=",")
                v[key] = np.loadtxt("special/NN/{}_{}_v_{}.csv"
                                    .format(i, string, key), delimiter=",")
            elif string[0] == 'Q':
                for k in ('r', 'g', 'b'):
                    key = j + k + str(l)
                    para[key] = np.loadtxt("special/NN/{}_{}_para_{}.csv"
                                           .format(string, key, i),
                                           delimiter=",")
                    h[key] = np.loadtxt("special/NN/{}_{}_h_{}.csv"
                                        .format(string, key, i), delimiter=",")
                    m[key] = np.loadtxt("special/NN/{}_{}_m_{}.csv"
                                        .format(i, string, key), delimiter=",")
                    v[key] = np.loadtxt("special/NN/{}_{}_v_{}.csv"
                                        .format(i, string, key), delimiter=",")
    NN.load_NN(para, h, m, v)

    return para, h, m, v


if __name__ == "__main__":
    main()
