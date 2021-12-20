from Gaussian import *
from Bayes import *
from Linear_Network import *
plt.rcParams["figure.figsize"] = (10.0, 10.0)


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


if __name__ == "__main__":
    mu_0 = np.array([-3.0, 1.0])  # [ D ]
    cov_0 = np.array([[1.0, 0.5], [0.5, 0.5]])  # [ D * D ]
    mu_1 = np.array([-2.0, -1.0])
    cov_1 = np.array([[3.0, 0.0], [0.0, 0.2]])
    mu_2 = np.array([-1.0, -3.0])
    cov_2 = np.array([[0.5, 0.5], [0.5, 1.0]])
    mu_3 = np.array([2.0, -1.0])
    cov_3 = np.array([[0.5, 0.0], [0.0, 2.0]])

    mu_set = np.array([mu_0, mu_1, mu_2, mu_3])  # [ K * D ]
    cov_set = np.array([cov_0, cov_1, cov_2, cov_3])  # [ K * D * D ]

    """ Initial Gaussian & Generate sample """

    gaussian_set = Gaussian_Set(mu_set, cov_set)
    sample_point, sample_label = gaussian_set.generate_point_set(N_k)
    gaussian_set.plot_gaussian_set(plot_confidence_interval=True)

    """ A. Bayes Inferences """

    bayes = Bayes(mu_set, cov_set)
    bayes_accuracy = bayes.bayes_accuracy(sample_point, sample_label)
    print("Bayes Inferences Accuracy: %10.7f" % bayes_accuracy)
    bayes.plot_decision_boundary(sample_point, plot_confidence_interval=True)

    """ B. GMM """

    """ C. Linear Neural Network """

    # get test sample
    gaussian_set_test = Gaussian_Set(mu_set, cov_set)
    test_point, test_label = gaussian_set_test.generate_point_set(N_k * 4)
    # train
    network = Linear_Network(D, neuron_num, activation_func, load_network=False)
    network.train(sample_point, sample_label, test_point, test_label,
                  train_number, gradient, optimizer, optimizer_para,
                  save_network=True)
    # plot train result
    network.plot_result(bayes_accuracy)
    network.plot_decision_boundary(sample_point, mu_set, cov_set,
                                   plot_confidence_interval=True)

    """ D. Quadratic Neural Network """
