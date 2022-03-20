import itertools
import numpy as np
import scipy.stats as st


class EM:
    """
    Example:
        em = EM(K)
        em.train(train_point)
        print("EM accuracy: %7.5f" % em.test(test_point, test_label))
    """

    def __init__(self, K):
        self.K = K      # number of clustering / Gaussian mixture
        self.D = None   # dimension, depend on the input point when training

        self.mu_set  = None     # mean of each mixture, [ K * D ]
        self.cov_set = None     # covariance of each mixture, [ K * D * D ]
        self.prio_p  = None     # prior probability of each mixture, [ K ]

    def E_step(self, point):
        """
        Get the posterior probability of input data point
        """
        post_p = np.zeros((len(point), self.K))     # [ N * K ]
        for k in range(self.K):
            post_p[:, k] = self.prio_p[k] * \
                           st.multivariate_normal.pdf(point,
                                                      self.mu_set[k],
                                                      self.cov_set[k],
                                                      allow_singular=True)
        post_p /= np.sum(post_p, axis=1)[:, None]   # [ N * K ]

        return post_p   # posterior probability of input data point, [ N * K ]

    def M_step(self, point, post_p):
        """
        Update prior probability, mu_set, cov_set according to "post_p" from E
        step.
        """
        # sum of posterior probability
        sum_post_p = np.sum(post_p, axis=0)     # [ K ]

        # 1. update prior probability
        self.prio_p = sum_post_p / len(point)   # [ K ]

        for k in range(self.K):
            below = sum_post_p[k]   # float

            # 2. update mu_set
            above = np.sum(point * post_p[:, [k]], axis=0)
            self.mu_set[k] = above / below

            # 3. update cov_set
            x_mu = point - self.mu_set[k]                       # [ N * D ]
            above = np.dot((post_p[:, [k]] * x_mu).T, x_mu)     # [ D * D ]
            self.cov_set[k] = above / below                     # [ D * D ]

        return self.mu_set, self.cov_set, self.prio_p

    def predict(self, point):
        return self.E_step(point)

    def order_correction(self, point, label):
        """
        The parameters that EM gets is correct but shuffled since we initialize
        mu of each Gaussian by random. We may get a bad accuracy result
        because the order of the parameters.
            This function will adjust the order of "mu_set," "cov_set," "prio_p"
        that maximize the accuracy.

        Args:
            point: [ sample_size * D ], np.array
            label: [ sample_size * K ], np.array

        Returns:
            accuracy, float
        """
        order = []  # [ K ], store the correct order
        accuracy = 0

        # get the correct order
        t = np.argmax(label, axis=1)
        for j in list(itertools.permutations([i for i in range(self.K)],
                                             self.K)):
            y = np.argmax(self.E_step(point)[:, j], axis=1)
            current_accuracy = np.sum(y == t) / len(label)
            if current_accuracy > accuracy:
                order = j
                accuracy = current_accuracy

        # change the order of mu, cov, prior probability according to the order
        for data in (self.mu_set, self.cov_set, self.prio_p):
            temp = np.copy(data)    # store the old data
            for i in range(self.K):
                data[i] = temp[order[i]]

        return accuracy

    def test(self, point, label, order_correction=True):
        """
        Test the accuracy using the current EM.
        If "order_correction" set as true, call the help function
        "self.order_correction" to correct the order of parameters. You can also
        call "self.order_correction" before test by yourself.
        """
        if self.prio_p is None: return 0    # means EM has not been trained

        if order_correction: return self.order_correction(point, label)

        t = np.argmax(label, axis=1)
        y = np.argmax(self.E_step(point), axis=1)

        return np.sum(y == t) / len(label)  # return the accuracy, float

    def train(self, train_point, epoch=2000, epsilon=1e-10):
        """
        Repeat E step and M step for "epoch" number of iteration.

        Args:
            train_point: [ sample_size * D ], np.array
            epoch: number of iteration, int
            epsilon: stop training when the norm of change of mu < epsilon
        """
        # initialize all parameters
        self.D       = len(train_point[0])
        self.mu_set  = np.random.randn(self.K, self.D)      # [ K * D ]
        self.cov_set = np.array([np.eye(self.D)] * self.K)  # [ K * D * D ]
        self.prio_p  = np.ones((self.K, 1)) / self.K        # [ K ]

        # train
        old_mu = self.mu_set.copy()
        for i in range(epoch):
            self.M_step(train_point, self.E_step(train_point))

            # breakpoint
            if np.linalg.norm(self.mu_set - old_mu) < epsilon: break
            old_mu = self.mu_set.copy()
