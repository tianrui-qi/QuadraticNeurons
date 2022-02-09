from visualization import *
from Gaussian import Gaussian
from EM import EM
from LNN import LNN
from QNN import QNN

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


''' Parameters '''

D   = 2         # dimension of sample data point
K   = 4         # number of Gaussian / classifications
N_k = 300       # number of sample for each Gaussian

run_number = 30
EM_train_number = 200
NN_train_number = 10000

neuron_num = {
    0: 10,
    1: K
}

LNN_activation_func = {
    0: LNN.relu,
    1: LNN.softmax
}
LNN_gradient  = LNN.gradient_bp  # gradient_ng, gradient_bp
LNN_optimizer = LNN.Adam         # SGD, AdaGrad, RMSprop, Adam

QNN_activation_func = {
    0: QNN.relu,
    1: QNN.softmax
}
QNN_gradient  = QNN.gradient_bp  # gradient_ng, gradient_bp
QNN_optimizer = QNN.Adam         # SGD, AdaGrad, RMSprop, Adam

optimizer_para = {
    "lr":         0.01,     # float, for all optimizer
    "decay_rate": 0.99,     # float, for optimizer "RMSprop"
    "beta1":      0.9,      # float, for optimizer "Adam"
    "beta2":      0.999,    # float, for optimizer "Adam"
    "iter":       0
}


''' Generate Sample '''
"""
mu_0 = np.array([-3.0, 1.0])  # [ D ]
cov_0 = np.array([[1.0, 0.5], [0.5, 0.5]])  # [ D * D ]
mu_1 = np.array([-1.0, -3.0])
cov_1 = np.array([[0.5, 0.5], [0.5, 1.0]])

mu_set = np.array([mu_0, mu_1])
cov_set = np.array([cov_0, cov_1])
"""
mu_0 = np.array([-3.0, 1.0])  # [ D ]
cov_0 = np.array([[1.0, 0.5], [0.5, 0.5]])  # [ D * D ]
mu_1 = np.array([-2.0, -1.0])
cov_1 = np.array([[3.0, 0.0], [0.0, 0.2]])
mu_2 = np.array([-1.0, -3.0])
cov_2 = np.array([[0.5, 0.5], [0.5, 1.0]])
mu_3 = np.array([2.0, -1.0])
cov_3 = np.array([[0.5, 0.0], [0.0, 2.0]])

mu_set = np.array([mu_0, mu_1, mu_2, mu_3])
cov_set = np.array([cov_0, cov_1, cov_2, cov_3])


train_point, train_label, test_point, test_label, \
sample_point, sample_label, = Gaussian(mu_set, cov_set). \
    generate_sample(N_k, load_sample=True, save_sample=True)

''' Visualization Parameters '''

edge = 1
x_max = sample_point[np.argmax(sample_point.T[0])][0] + edge
x_min = sample_point[np.argmin(sample_point.T[0])][0] - edge
y_max = sample_point[np.argmax(sample_point.T[1])][1] + edge
y_min = sample_point[np.argmin(sample_point.T[1])][1] - edge
# plt.rcParams["figure.figsize"] = (10.0, 10.0)
color = ("blue", "orange", "red", "green")
legend = [mpatches.Patch(color=color[i], label="Gaussian_{}".format(i))
          for i in range(K)]
# legend.append(mpatches.Patch(color=color[K-1], label="Background"))


''' Main '''

def main():
    # Plot Samples
    plot_sample()
    
    # A. Expectation Maximization (EM)
    em = EM()
    accuracy = 0
    while accuracy < 0.8:
        em.train(train_point, train_label, EM_train_number)
        accuracy = em.test(test_point, test_label)
    print("EM accuracy: %7.5f" % accuracy)
    plot_EM_DB(em)

    """
    ''' B. Linear Neural Network (LNN) '''

    for i in range(run_number):     # test LNN for "run_number" time
        lnn = LNN(D, neuron_num, LNN_activation_func, load_LNN=False)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, LNN_gradient, LNN_optimizer, optimizer_para,
                  save_LNN=True, save_result=i)
        plot_LNN_DB(lnn, i)
    average_LNN()

    ''' C. Quadratic Neural Network (QNN) '''
    
    for i in range(run_number):  # test QNN for "run_number" time
        qnn = QNN(D, neuron_num, QNN_activation_func, load_QNN=False)
        qnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, QNN_gradient, QNN_optimizer, optimizer_para,
                  save_QNN=True, save_result=i)
        plot_QNN_DB(qnn, i)
    average_QNN()
    """


''' Help Function For main '''

def plot_sample():
    fig, ax = plt.subplots()
    plot_scatter(sample_point, sample_label, ax, color)
    plot_confidence_interval_fill(mu_set, cov_set, ax, color)
    plt.legend(handles=legend)
    plt.title("Sample Point", fontsize=14)
    plt.axis([x_min, x_max, y_min, y_max])
    plt.grid()
    fig.show()
    fig.savefig("sample.png")


def plot_EM_DB(em):
    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(mu_set, cov_set, ax, color)
    plot_decision_boundary(K, em.predict,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("Expectation Maximization (EM) Decision Boundary",
              fontsize=14)
    plt.axis([x_min, x_max, y_min, y_max])
    plt.grid()
    fig.show()

    fig.savefig("EM_DB.png")


def plot_LNN_DB(lnn, i):
    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(mu_set, cov_set, ax, color)
    plot_decision_boundary(K, lnn.predict,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("Linear Neural Network (LNN) Decision Boundary", fontsize=14)
    plt.axis([x_min, x_max, y_min, y_max])
    plt.grid()
    fig.show()

    if not os.path.exists('LNN_result'): os.mkdir('LNN_result')
    fig.savefig("LNN_result/DB_{}.png".format(i))


def plot_QNN_DB(qnn, i):
    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(mu_set, cov_set, ax, color)
    plot_decision_boundary(K, qnn.predict,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("Quadratic Neural Network (QNN) Decision Boundary",
              fontsize=14)
    plt.axis([x_min, x_max, y_min, y_max])
    plt.grid()
    fig.show()

    if not os.path.exists('QNN_result'): os.mkdir('QNN_result')
    fig.savefig("QNN_result/DB_{}.png".format(i))


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
