from visualization import *
from Gaussian import Gaussian
from Bayes import Bayes
from EM import EM
from Linear_Network import Linear_Network

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

D   = 2         # dimension of sample data point
K   = 4         # number of Gaussian / classifications
N_k = 10000     # number of sample for each Gaussian


neuron_num = {
    0: 100,
    1: 100,
    2: 100,
    3: 100,
    4: K
}
activation_func = {
    0: Linear_Network.sigmoid,
    1: Linear_Network.sigmoid,
    2: Linear_Network.sigmoid,
    3: Linear_Network.sigmoid,
    4: Linear_Network.softmax
}

train_number = 500
gradient  = Linear_Network.gradient_bp     # gradient_ng, gradient_bp
optimizer = Linear_Network.Adam         # SGD, AdaGrad, RMSprop, Adam
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

    """ Generate Samples """

    train_point, train_label, test_point, test_label, \
    sample_point, sample_label, = Gaussian(mu_set, cov_set).\
        generate_sample(N_k, load_sample=True, save_sample=True)

    """ Visualization Parameters """

    x_max = sample_point[np.argmax(sample_point.T[0])][0]
    x_min = sample_point[np.argmin(sample_point.T[0])][0]
    y_max = sample_point[np.argmax(sample_point.T[1])][1]
    y_min = sample_point[np.argmin(sample_point.T[1])][1]
    # plt.rcParams["figure.figsize"] = (10.0, 10.0)
    color  = ("blue", "orange", "green", "red")
    legend = [mpatches.Patch(color=color[i], label="Gaussian_{}".format(i))
              for i in range(K) ]

    """ Plot Samples """
    
    fig, ax = plt.subplots()
    plot_scatter(sample_point, sample_label, ax, color)
    plot_confidence_interval_fill(mu_set, cov_set, ax, color)
    plt.legend(handles=legend)
    plt.title("Gaussian Sample Point")
    plt.axis([x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3])
    plt.grid()
    fig.show()

    """ Bayes Inferences """

    bayes = Bayes(mu_set, cov_set)
    bayes_accuracy = bayes.accuracy(train_point, train_label)
    print("Bayes Inferences Accuracy: %10.7f" % bayes_accuracy)

    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(mu_set, cov_set, ax, color)
    plot_decision_boundary(K, bayes.inferences,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("Bayes Inferences Decision Boundary")
    plt.axis([x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3])
    plt.grid()
    fig.show()

    """ 1. EM """

    em = EM()
    em_accuracy = em.train(train_point, train_label, test_point, test_label,
                           train_number, save_EM=True)

    fig, ax = plt.subplots()
    plot_confidence_interval_unfill(em.mu_set, em.cov_set, ax, color)
    plot_decision_boundary(K, Bayes(em.mu_set, em.cov_set).inferences,
                           ax, color, x_min, x_max, y_min, y_max)
    plt.legend(handles=legend)
    plt.title("EM Decision Boundary")
    plt.axis([x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3])
    plt.grid()
    fig.show()

    """ 2. Linear Neural Network """

    network = Linear_Network(D, neuron_num, activation_func, load_network=False)
    network.train(train_point, train_label, test_point, test_label,
                  train_number, gradient, optimizer, optimizer_para,
                  save_network=True)

    """ 3. Quadratic Neural Network """
