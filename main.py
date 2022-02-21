from Gaussian import Gaussian
from EM import EM
from LNN import LNN
from QNN import QNN
from Visual import Visual

import numpy as np

D   = 2         # dimension of sample data point
K   = 6        # number of Gaussian / classifications

# number of sample for each Gaussian
N_k = [np.random.randint(3000, 6000) for k in range(K)]
N = np.sum(N_k)

run_number = 1
EM_train_number = 2000
NN_train_number = 10000

LNN_neuron_num = {
    0: K
}
LNN_activation_func = {
    0: LNN.softmax
}

QNN_neuron_num      = { 0: K }
QNN_activation_func = { 0: QNN.softmax }

optimizer_para = {
    "lr":         0.01,     # float, for all optimizer
    "decay_rate": 0.99,     # float, for optimizer "RMSprop"
    "beta1":      0.9,      # float, for optimizer "Adam"
    "beta2":      0.999,    # float, for optimizer "Adam"
    "iter":       0
}


def main():
    # Generate Sample
    mu_set = np.array([(np.random.random(D) - 0.5) * 15 for i in range(K)])
    cov_set = []
    for i in range(K):
        a = np.random.random((D, D)) * 2 - 1
        cov = np.dot(a, a.T) + np.dot(a, a.T)
        cov_set.append(cov)

    train_point, train_label, test_point, test_label, \
    sample_point, sample_label, = Gaussian(mu_set, cov_set). \
        generate_sample(N_k)

    # Initializing Visualization
    visual = Visual(sample_point, sample_label, mu_set, cov_set)

    # Plot Samples
    visual.plot_sample()

    # A. Expectation Maximization (EM)
    em = EM(K)
    em.train(train_point, train_label, EM_train_number)
    print("EM accuracy: %7.5f" % em.test(test_point, test_label))
    # visual.plot_EM_DB(em)


    for i in range(run_number):
        # B. LNN
        lnn = LNN(D, LNN_neuron_num, LNN_activation_func)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)

        # C. QNN
        qnn = QNN(D, QNN_neuron_num, QNN_activation_func)
        qnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, optimizer_para)


def average_LNN():
    train_accuracy = np.zeros([NN_train_number+1])
    test_accuracy  = np.zeros([NN_train_number+1])
    train_loss     = np.zeros([NN_train_number+1])
    test_loss      = np.zeros([NN_train_number+1])
    for i in range(run_number):
        train_accuracy += np.loadtxt("LNN_result/train_accuracy_{}.csv"
                                     .format(i), delimiter=",")
        test_accuracy  += np.loadtxt("LNN_result/test_accuracy_{}.csv"
                                     .format(i), delimiter=",")
        train_loss     += np.loadtxt("LNN_result/train_loss_{}.csv"
                                     .format(i), delimiter=",")
        test_loss      += np.loadtxt("LNN_result/test_loss_{}.csv"
                                     .format(i), delimiter=",")
    np.savetxt("LNN_result/train_accuracy_average.csv",
               train_accuracy/float(run_number), delimiter=",")
    np.savetxt("LNN_result/test_accuracy_average.csv",
               test_accuracy/float(run_number), delimiter=",")
    np.savetxt("LNN_result/train_loss_average.csv",
               train_loss/float(run_number), delimiter=",")
    np.savetxt("LNN_result/test_loss_average.csv",
               test_loss/float(run_number), delimiter=",")


def average_QNN():
    train_accuracy = np.zeros([NN_train_number+1])
    test_accuracy  = np.zeros([NN_train_number+1])
    train_loss     = np.zeros([NN_train_number+1])
    test_loss      = np.zeros([NN_train_number+1])
    for i in range(run_number):
        train_accuracy += np.loadtxt("QNN_result/train_accuracy_{}.csv"
                                     .format(i), delimiter=",")
        test_accuracy  += np.loadtxt("QNN_result/test_accuracy_{}.csv"
                                     .format(i), delimiter=",")
        train_loss     += np.loadtxt("QNN_result/train_loss_{}.csv"
                                     .format(i), delimiter=",")
        test_loss      += np.loadtxt("QNN_result/test_loss_{}.csv"
                                     .format(i), delimiter=",")
    np.savetxt("QNN_result/train_accuracy_average.csv",
               train_accuracy/float(run_number), delimiter=",")
    np.savetxt("QNN_result/test_accuracy_average.csv",
               test_accuracy/float(run_number), delimiter=",")
    np.savetxt("QNN_result/train_loss_average.csv",
               train_loss/float(run_number), delimiter=",")
    np.savetxt("QNN_result/test_loss_average.csv",
               test_loss/float(run_number), delimiter=",")


if __name__ == "__main__":
    main()
