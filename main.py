from visualization import *
from Gaussian import Gaussian
from Bayes import Bayes
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
N_k = 15000     # number of sample for each Gaussian

run_number = 30
EM_train_number = 100
NN_train_number = 400

neuron_num = {
    0: 100,
    1: 100,
    2: 100,
    3: 100,
    4: K
}

LNN_activation_func = {
    0: LNN.sigmoid,
    1: LNN.sigmoid,
    2: LNN.sigmoid,
    3: LNN.sigmoid,
    4: LNN.softmax
}
LNN_gradient  = LNN.gradient_bp  # gradient_ng, gradient_bp
LNN_optimizer = LNN.Adam         # SGD, AdaGrad, RMSprop, Adam

QNN_activation_func = {
    0: QNN.sigmoid,
    1: QNN.sigmoid,
    2: QNN.sigmoid,
    3: QNN.sigmoid,
    4: QNN.softmax
}
QNN_gradient  = QNN.gradient_bp  # gradient_ng, gradient_bp
QNN_optimizer = QNN.Adam         # SGD, AdaGrad, RMSprop, Adam

optimizer_para = {
    "lr":         0.01,    # float, for all optimizer
    "decay_rate": 0.99,     # float, for optimizer "RMSprop"
    "beta1":      0.9,      # float, for optimizer "Adam"
    "beta2":      0.999,     # float, for optimizer "Adam"
    "iter":       0
}

''' Generate Sample '''

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

x_max = sample_point[np.argmax(sample_point.T[0])][0]
x_min = sample_point[np.argmin(sample_point.T[0])][0]
y_max = sample_point[np.argmax(sample_point.T[1])][1]
y_min = sample_point[np.argmin(sample_point.T[1])][1]
# plt.rcParams["figure.figsize"] = (10.0, 10.0)
color = ("blue", "orange", "green", "red")
legend = [mpatches.Patch(color=color[i], label="Gaussian_{}".format(i))
          for i in range(K)]


def plot_sample():
    fig, ax = plt.subplots()
    plot_scatter(sample_point, sample_label, ax, color)
    plot_confidence_interval_fill(mu_set, cov_set, ax, color)
    plt.legend(handles=legend)
    plt.title("Sample Point", fontsize=14)
    plt.axis([x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5])
    plt.grid()
    fig.show()
    fig.savefig("sample.png")


def plot_Bayes_DB(bayes):
    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(mu_set, cov_set, ax, color)
    plot_decision_boundary(K, bayes.predict,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("Bayes Inferences Decision Boundary", fontsize=14)
    plt.axis([x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5])
    plt.margins(0, 0)
    plt.grid()
    fig.show()
    fig.savefig("Bayes_DB.png")


def plot_EM_DB(em, i):
    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(mu_set, cov_set, ax, color)
    plot_decision_boundary(K, Bayes(em.mu_set, em.cov_set).predict,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("Expectation Maximization (EM) Decision Boundary",
              fontsize=14)
    plt.axis([x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5])
    plt.grid()
    fig.show()

    if not os.path.exists('EM_result'): os.mkdir('EM_result')
    fig.savefig("EM_result/DB_{}.png".format(i))


def plot_LNN_DB(lnn, i):
    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(mu_set, cov_set, ax, color)
    plot_decision_boundary(K, lnn.predict,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("Linear Neural Network (LNN) Decision Boundary", fontsize=14)
    plt.axis([x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5])
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
    plt.axis([x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5])
    plt.grid()
    fig.show()

    if not os.path.exists('LNN_result'): os.mkdir('LNN_result')
    fig.savefig("LNN_result/DB_{}.png".format(i))


def plot_EM_accuracy():
    if not os.path.exists('EM_result'): return
    fig, ax = plt.subplots()
    for i in range(run_number):
        ax.plot(np.loadtxt("EM_result/accuracy_{}.csv".format(i),
                           delimiter=","))
    plt.title("Expectation Maximization (EM) Accuracy", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.grid()
    fig.show()

    if not os.path.exists('EM_result'): os.mkdir('EM_result')
    fig.savefig("EM_result/accuracy.png")


def plot_LNN_accuracy(bayes_accuracy):
    # train accuracy

    fig, ax = plt.subplots()
    plt.plot(bayes_accuracy + np.zeros(NN_train_number),
             color="red", linewidth=3)
    for i in range(run_number):
        ax.plot(np.loadtxt("LNN_result/train_accuracy_{}.csv".format(i),
                           delimiter=","), color="green", linewidth=1)
    plt.title("Linear Neural Network (LNN) Train Accuracy", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(["Bayes Accuracy", "Train Accuracy"])
    plt.ylim(0, 1)
    plt.grid()
    fig.show()

    # train accuracy (detail)

    fig, ax = plt.subplots()
    plt.plot(bayes_accuracy + np.zeros(NN_train_number),
             color="red", linewidth=3)
    for i in range(run_number):
        ax.plot(np.loadtxt("LNN_result/train_accuracy_{}.csv".format(i),
                           delimiter=","), color="green", linewidth=1)
    plt.title("Linear Neural Network (LNN) Train Accuracy (detail)",
              fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(["Bayes Accuracy", "Train Accuracy"])
    plt.ylim(bayes_accuracy - 0.015, bayes_accuracy + 0.005)
    plt.grid()
    fig.show()

    # test accuracy

    fig, ax = plt.subplots()
    plt.plot(bayes_accuracy + np.zeros(NN_train_number),
             color="red", linewidth=3)
    for i in range(run_number):
        ax.plot(np.loadtxt("LNN_result/test_accuracy_{}.csv".format(i),
                           delimiter=","), color="blue", linewidth=1)
    plt.title("Linear Neural Network (LNN) Test Accuracy", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(["Bayes Accuracy", "Test Accuracy"])
    plt.ylim(0, 1)
    plt.grid()
    fig.show()

    # test accuracy (detail)

    fig, ax = plt.subplots()
    plt.plot(bayes_accuracy + np.zeros(NN_train_number),
             color="red", linewidth=3)
    for i in range(run_number):
        ax.plot(np.loadtxt("LNN_result/test_accuracy_{}.csv".format(i),
                           delimiter=","), color="blue", linewidth=1)
    plt.title("Linear Neural Network (LNN) Test Accuracy (detail)",
              fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(["Bayes Accuracy", "Test Accuracy"])
    plt.ylim(bayes_accuracy - 0.015, bayes_accuracy + 0.005)
    plt.grid()
    fig.show()


if __name__ == "__main__":
    ''' Plot Samples '''

    plot_sample()

    ''' Bayes Inferences '''

    bayes = Bayes(mu_set, cov_set)
    bayes_accuracy = bayes.accuracy(test_point, test_label)
    print("Bayes Inferences Accuracy: %10.7f" % bayes_accuracy)
    plot_Bayes_DB(bayes)
    
    ''' A. Expectation Maximization (EM) '''

    for i in range(run_number):     # test EM for "run_number" time
        em = EM()
        em.train(train_point, train_label, test_point, test_label,
                 EM_train_number, save_result=i)
        plot_EM_DB(em, i)
    plot_EM_accuracy()

    ''' B. Linear Neural Network (LNN) '''

    for i in range(run_number):     # test LNN for "run_number" time
        lnn = LNN(D, neuron_num, LNN_activation_func, load_LNN=False)
        lnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, LNN_gradient, LNN_optimizer, optimizer_para,
                  save_LNN=True, save_result=i)
        plot_LNN_DB(lnn, i)
    plot_LNN_accuracy(bayes_accuracy)

    ''' C. Quadratic Neural Network (QNN) '''

    for i in range(run_number):     # test QNN for "run_number" time
        qnn = QNN(D, neuron_num, QNN_activation_func, load_QNN=False)
        qnn.train(train_point, train_label, test_point, test_label,
                  NN_train_number, QNN_gradient, QNN_optimizer, optimizer_para,
                  save_QNN=True, save_result=i)
        plot_QNN_DB(qnn, i)
