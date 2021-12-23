from visualization import *
from Gaussian import Gaussian
from Bayes import Bayes
from EM import EM
from LNN import LNN
from QNN import QNN

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

D   = 2         # dimension of sample data point
K   = 4         # number of Gaussian / classifications
N_k = 15000     # number of sample for each Gaussian

neuron_num = {
    0: 100,
    1: 100,
    2: 100,
    3: 100,
    4: K
}
train_number = 300

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


def init_mu_cov():
    """
    :return: mu_set:  [ K * D ], np.array
             cov_set: [ K * D * D ], np.array
    """
    mu_0 = np.array([-3.0, 1.0])                # [ D ]
    cov_0 = np.array([[1.0, 0.5], [0.5, 0.5]])  # [ D * D ]
    mu_1 = np.array([-2.0, -1.0])
    cov_1 = np.array([[3.0, 0.0], [0.0, 0.2]])
    mu_2 = np.array([-1.0, -3.0])
    cov_2 = np.array([[0.5, 0.5], [0.5, 1.0]])
    mu_3 = np.array([2.0, -1.0])
    cov_3 = np.array([[0.5, 0.0], [0.0, 2.0]])

    return np.array([mu_0, mu_1, mu_2, mu_3]), \
           np.array([cov_0, cov_1, cov_2, cov_3])


if __name__ == "__main__":
    mu_set, cov_set = init_mu_cov()

    ''' 
    Generate Samples 
    '''

    train_point, train_label, test_point, test_label, \
    sample_point, sample_label, = Gaussian(mu_set, cov_set).\
        generate_sample(N_k, load_sample=True, save_sample=True)

    ''' 
    Visualization Parameters 
    '''

    x_max = sample_point[np.argmax(sample_point.T[0])][0]
    x_min = sample_point[np.argmin(sample_point.T[0])][0]
    y_max = sample_point[np.argmax(sample_point.T[1])][1]
    y_min = sample_point[np.argmin(sample_point.T[1])][1]
    # plt.rcParams["figure.figsize"] = (10.0, 10.0)
    color  = ("blue", "orange", "green", "red")
    legend = [mpatches.Patch(color=color[i], label="Gaussian_{}".format(i))
              for i in range(K) ]

    ''' 
    Plot Samples 
    '''

    fig, ax = plt.subplots()
    plot_scatter(sample_point, sample_label, ax, color)
    plot_confidence_interval_fill(mu_set, cov_set, ax, color)
    plt.legend(handles=legend)
    plt.title("Gaussian Sample Point")
    plt.axis([x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3])
    plt.grid()
    fig.show()
    
    ''' 
    Bayes Inferences 
    '''

    bayes = Bayes(mu_set, cov_set)
    bayes_accuracy = bayes.accuracy(train_point, train_label)
    print("Bayes Inferences Accuracy: %10.7f" % bayes_accuracy)

    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(mu_set, cov_set, ax, color)
    plot_decision_boundary(K, bayes.predict,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("Bayes Inferences Decision Boundary")
    plt.axis([x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3])
    plt.grid()
    fig.show()
    
    ''' 
    A. Expectation Maximization (EM) 
    '''
    
    em = EM()
    em_accuracy = em.train(train_point, train_label, test_point, test_label,
                           train_number, save_EM=True)

    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(mu_set, cov_set, ax, color)
    plot_decision_boundary(K, Bayes(em.mu_set, em.cov_set).predict,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("Expectation Maximization (EM) Decision Boundary")
    plt.axis([x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3])
    plt.grid()
    fig.show()
    
    ''' 
    B. Linear Neural Network (LNN) 
    '''

    lnn = LNN(D, neuron_num, LNN_activation_func, load_LNN=False)
    lnn_train_accuracy, lnn_test_accuracy = \
        lnn.train(train_point, train_label, test_point, test_label,
                  train_number, LNN_gradient, LNN_optimizer, optimizer_para,
                  save_LNN=True)

    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(mu_set, cov_set, ax, color)
    plot_decision_boundary(K, lnn.predict,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("Linear Neural Network (LNN) Decision Boundary")
    plt.axis([x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3])
    plt.grid()
    fig.show()

    ''' 
    C. Quadratic Neural Network (QNN) 
    '''

    qnn = QNN(D, neuron_num, QNN_activation_func, load_QNN=False)
    qnn_train_accuracy, qnn_test_accuracy = \
        qnn.train(train_point, train_label, test_point, test_label,
                  train_number, QNN_gradient, QNN_optimizer, optimizer_para,
                  save_QNN=True)

    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(mu_set, cov_set, ax, color)
    plot_decision_boundary(K, qnn.predict,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("Quadratic Neural Network (QNN) Decision Boundary")
    plt.axis([x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3])
    plt.grid()
    fig.show()

    """
    def plot_result(self, bayes_accuracy):
        # plot the accuracy cure
        plt.plot(bayes_accuracy + np.zeros_like(self.train_accuracy_set),
                 color="blue")
        plt.plot(self.train_accuracy_set, color="red")
        plt.plot(self.test_accuracy_set, color="green")

        plt.legend(["Bayes", "Linear NN (train)", "Linear NN (test)"],
                   fontsize=14)
        plt.title("Linear Neural Network Accuracy (Detail)", fontsize=14)
        plt.xlabel("Train Number", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.ylim(0, 1)
        plt.grid()
        plt.show()

        # plot the detail accuracy cure
        plt.plot(bayes_accuracy + np.zeros_like(self.train_accuracy_set),
                 linewidth=2, color="blue")
        plt.plot(self.train_accuracy_set, color="red")
        plt.plot(self.test_accuracy_set, color="green")

        plt.legend(["Bayes", "Linear NN (train)", "Linear NN (test)"],
                   fontsize=14)
        plt.title("Linear Neural Network Accuracy (Detail)", fontsize=14)
        plt.xlabel("Train Number", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.ylim(bayes_accuracy-0.015, bayes_accuracy+0.005)
        plt.grid()
        plt.show()
    """
