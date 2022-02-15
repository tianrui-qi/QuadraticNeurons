from Gaussian import Gaussian
from EM import EM
from LNN import LNN
from QNN import QNN
from visualization import Visual

import numpy as np

bg = True

D   = 2         # dimension of sample data point
K   = 2         # number of Gaussian / classifications
N_k = 1500      # number of sample for each Gaussian

run_number = 1
EM_train_number = 500
NN_train_number = 10000

LNN_neuron_num = {
    0: K + 1 * bg
}
LNN_activation_func = {
    0: LNN.softmax
}
LNN_gradient  = LNN.gradient_bp  # gradient_ng, gradient_bp
LNN_optimizer = LNN.Adam          # SGD, AdaGrad, RMSprop, Adam

QNN_neuron_num      = { 0: K + 1 * bg }
QNN_activation_func = { 0: QNN.softmax }
QNN_gradient  = QNN.gradient_bp  # gradient_ng, gradient_bp
QNN_optimizer = QNN.Adam          # SGD, AdaGrad, RMSprop, Adam

optimizer_para = {
    "lr":         0.01,     # float, for all optimizer
    "decay_rate": 0.99,     # float, for optimizer "RMSprop"
    "beta1":      0.9,      # float, for optimizer "Adam"
    "beta2":      0.999,    # float, for optimizer "Adam"
    "iter":       0
}


def main():
    # Generate Sample
    if bg:
        mu_0 = np.array([-3.0, 1.0])  # [ D ]
        cov_0 = np.array([[1.0, 0.5], [0.5, 0.5]])  # [ D * D ]
        mu_1 = np.array([-1.0, -3.0])
        cov_1 = np.array([[0.5, 0.5], [0.5, 1.0]])

        mu_set = np.array([mu_0, mu_1])
        cov_set = np.array([cov_0, cov_1])
    else:
        mu_set = np.array([(np.random.random(D) - 0.5) * 10 for i in range(K)])
        cov_set = []
        for i in range(K):
            a = np.random.random((D, D)) * 2 - 1
            cov = np.dot(a, a.T) + np.dot(a, a.T)
            cov_set.append(cov)

    train_point, train_label, test_point, test_label, \
    sample_point, sample_label, = Gaussian(mu_set, cov_set, bg=bg). \
        generate_sample(N_k, load_sample=False, save_sample=False)

    # Initializing Visualization
    visual = Visual(sample_point, sample_label, mu_set, cov_set, bg=bg)

    # Plot Samples
    visual.plot_sample()

    # A. Expectation Maximization (EM)
    if not bg:
        em = EM(K)
        em.train(train_point, EM_train_number)
        print("EM accuracy: %7.5f" % em.test(test_point, test_label))
        visual.plot_EM_DB(em)

    # B. Linear Neural Network (LNN) '''
    for i in range(run_number):
        lnn = LNN(D, LNN_neuron_num, LNN_activation_func, load_LNN=False)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, LNN_gradient, LNN_optimizer, optimizer_para,
                  save_LNN=True, save_result=i)
        visual.plot_LNN_DB(lnn, i)
    # average_LNN()

    # C. Quadratic Neural Network (QNN) '''
    for i in range(run_number):
        qnn = QNN(D, QNN_neuron_num, QNN_activation_func, load_QNN=False)
        qnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, QNN_gradient, QNN_optimizer, optimizer_para,
                  save_QNN=True, save_result=i)
        visual.plot_QNN_DB(qnn, i)
    # average_QNN()


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
