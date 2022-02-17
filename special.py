from Gaussian import Gaussian
from LNN import LNN
from QNN import QNN
from Visual import Visual

import os
import numpy as np


D   = 2     # dimension of sample data point
K   = 3     # number of Gaussian / classifications
N_k = [np.random.randint(1000, 2000) for k in range(K-1)]

mu_set  = np.array([[-3.0, 1.0],              [-1.0, -3.0]])
cov_set = np.array([[[1.0, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 1.0]]])

run_number = 10
NN_train_number = 10000

LNN_neuron_num_1     = { 0: K }
LNN_neuron_num_2_10  = { 0: 10, 1: K }
LNN_neuron_num_2_50  = { 0: 50, 1: K }
LNN_neuron_num_2_100 = { 0: 100, 1: K }
LNN_neuron_num_3_50  = { 0: 50, 1: 50, 2: K }
QNN_neuron_num_1     = { 0: K }

LNN_activation_func_1 = { 0: LNN.softmax }
LNN_activation_func_2 = { 0: LNN.relu, 1: LNN.softmax }
LNN_activation_func_3 = { 0: LNN.relu, 1: LNN.relu, 2: LNN.softmax }
QNN_activation_func_1 = { 0: QNN.softmax }

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
    if not os.path.exists('special/save'): os.mkdir('special/save')
    if not os.path.exists('special/result'): os.mkdir('special/result')
    if not os.path.exists('special/sample'): os.mkdir('special/sample')

    gaussian = Gaussian(mu_set, cov_set)
    train_point, train_label, test_point, test_label, \
    sample_point, sample_label, = gaussian.generate_sample(N_k, bg=True)

    file = open("special/sample/parameter.txt", "w")
    file.write("K = {}, D = {}\nN_k = {}\n\n".format(K, D, gaussian.N_k))
    for i in range(K-1):
        file.write("Gaussian {}\nmu:\n  {}\ncov:\n  {}\n\n"
                   .format(i, gaussian.mu_set[i], gaussian.cov_set[i]))
    file.close()
    np.savetxt("special/sample/sample_point.csv", sample_point, delimiter=",")
    np.savetxt("special/sample/sample_label.csv", sample_label, delimiter=",")
    np.savetxt("special/sample/train_point.csv", train_point, delimiter=",")
    np.savetxt("special/sample/train_label.csv", train_label, delimiter=",")
    np.savetxt("special/sample/test_point.csv", test_point, delimiter=",")
    np.savetxt("special/sample/test_label.csv", test_label, delimiter=",")

    visual = Visual(sample_point, sample_label, mu_set, cov_set, bg=True)
    visual.plot_sample().savefig("special/fig/sample")

    for i in range(run_number):
        string = "L({}-{})".format(D, K)
        lnn = LNN(D, LNN_neuron_num_1, LNN_activation_func_1)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(lnn, i, string)
        visual.plot_LNN_DB(lnn).savefig("special/fig/{}_{}".format(string, i))

        string = "L({}-10-{})".format(D, K)
        lnn = LNN(D, LNN_neuron_num_2_10, LNN_activation_func_2)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(lnn, i, string)
        visual.plot_LNN_DB(lnn).savefig("special/fig/{}_{}".format(string, i))

        string = "L({}-50-{})".format(D, K)
        lnn = LNN(D, LNN_neuron_num_2_50, LNN_activation_func_2)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(lnn, i, string)
        visual.plot_LNN_DB(lnn).savefig("special/fig/{}_{}".format(string, i))

        string = "L({}-100-{})".format(D, K)
        lnn = LNN(D, LNN_neuron_num_2_100, LNN_activation_func_2)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(lnn, i, string)
        visual.plot_LNN_DB(lnn).savefig("special/fig/{}_{}".format(string, i))

        string = "L({}-50-50-{})".format(D, K)
        lnn = LNN(D, LNN_neuron_num_3_50, LNN_activation_func_3)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(lnn, i, string)
        visual.plot_LNN_DB(lnn).savefig("special/fig/{}_{}".format(string, i))

        string = "Q({}-{})".format(D, K)
        qnn = QNN(D, QNN_neuron_num_1, QNN_activation_func_1)
        qnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)
        save_NN(qnn, i, string)
        visual.plot_QNN_DB(qnn).savefig("special/fig/{}_{}".format(string, i))


def save_NN(NN, i, string):
    np.savetxt("special/result/{}_train_loss_{}.csv".format(string, i),
               NN.train_loss, delimiter=",")
    np.savetxt("special/result/{}_test_loss_{}.csv".format(string, i),
               NN.test_loss, delimiter=",")
    np.savetxt("special/result/{}_train_accuracy_{}.csv".format(string, i),
               NN.train_accuracy, delimiter=",")
    np.savetxt("special/result/{}_test_accuracy_{}.csv".format(string, i),
               NN.test_accuracy, delimiter=",")

    for key in NN.para.keys():
        np.savetxt("special/save/{}_para_{}_{}.csv".format(string, key, i),
                   NN.para[key], delimiter=",")
    for key in NN.h.keys():
        np.savetxt("special/save/{}_h_{}_{}.csv".format(string, key, i),
                   NN.h[key], delimiter=",")
    for key in NN.m.keys():
        np.savetxt("special/save/{}_m_{}_{}.csv".format(string, key, i),
                   NN.m[key], delimiter=",")
    for key in NN.v.keys():
        np.savetxt("special/save/{}_v_{}_{}.csv".format(string, key, i),
                   NN.v[key], delimiter=",")


if __name__ == "__main__":
    main()
