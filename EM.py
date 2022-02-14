import numpy as np
import scipy.stats as st


class EM:
    """
    Example:

        em = EM(K)
        em.train(train_point, EM_train_number)
        print("EM accuracy: %7.5f" % em.test(test_point, test_label))

    Notes that code above may not get good accuracy since the "test_label" is
    ordered by cluster. However, the order that EM get is shuffled since we
    initialize mu of each Gaussian by random.
    One way to solve this problem is:

        em = EM(K)
        while em.test(test_point, test_label) < 0.8:
            em.train(train_point, EM_train_number)
        print("EM accuracy: %7.5f" % em.test(test_point, test_label))

    If the train result is not good, means the order of result cluster is not
    right. Then, use "em.train" to re-initialize mu and re-train. The tol 0.8
    can be adjusted according to the problem.
    """

    def __init__(self, K):
        """
        :param K: number of clustering / Gaussian mixture, int
        """
        self.K = K
        self.D = None

        self.mu_set  = None     # [ K * D ]
        self.cov_set = None     # [ K * D * D ]
        self.prio_p  = None     # [ K ]

    """ Trainer """

    def E_step(self, sample_point):
        """
        The goal here is in fact to find the posterior probability. Recall the
        Bayes formula, we have that
                f(para|x): posterior probability
                f(x|para): likelihood
                f(para)  : prior probability
                f(x)     : marginal likelihood - total probability
            f(para|x) = f(para) * f(x|para) / f(x)
            post_p    = prio_p  * l         / marg_l
            [ N * K ] = [ K ]   * [ N * K ] / [ N ]
        By train the EM, we get predict mu, covariance, and prior probability of
        each cluster / Gaussian mixture. Then, we predict the cluster that each
        point of input "sample_point" belongs to. This function will return the
        posterior probability.

        :param sample_point: [ sample_size * D ], np.array
        :return: posterior probability, [ sample_size * K ], np.array
        """
        post_p = np.zeros((len(sample_point), self.K))  # [ N * K ]
        for k in range(self.K):
            post_p[:, k] = self.prio_p[k] * \
                           st.multivariate_normal.pdf(sample_point,
                                                      self.mu_set[k],
                                                      self.cov_set[k])

        return post_p/np.sum(post_p, axis=1)[:, None]   # [ N * K ]

    def M_step(self, sample_point, post_p):
        """
        Update prior probability, mu_set, cov_set according to "post_p"

        :param sample_point: [ sample_size * D ], np.array
        :param post_p: posterior probability, [ N * K ], np.array
        :return self.mu_set: [ K * D ], np.array
        :return self.cov_set: [ K * D * D ], np.array
        :return self.prio_p: prior probability, [ K ], np.array
        """
        sum_post_p = np.sum(post_p, axis=0)             # [ K ]

        # update prior probability
        self.prio_p = sum_post_p / len(sample_point)    # [ K ]

        for k in range(self.K):
            below = sum_post_p[k]   # float

            # update mu_set
            above = np.sum(sample_point * post_p[:, [k]], axis=0)
            self.mu_set[k] = above / below

            # update cov_set
            x_mu = sample_point - self.mu_set[k]                # [ N * D ]
            above = np.dot((post_p[:, [k]] * x_mu).T, x_mu)     # [ D * D ]
            self.cov_set[k] = above / below                     # [ D * D ]

        return self.mu_set, self.cov_set, self.prio_p

    def train(self, train_point, train_number):
        """
        Repeat E step and M step for "train_number" time.

        :param train_point: [ sample_size * D ], np.array
        :param train_number: number of iteration
        :return self.mu_set: [ K * D ], np.array
        :return self.cov_set: [ K * D * D ], np.array
        :return self.prio_p: prior probability, [ K ], np.array
        """
        self.D = len(train_point[0])
        self.mu_set  = np.random.randn(self.K, self.D)      # [ K * D ]
        self.cov_set = np.array([np.eye(self.D)] * self.K)  # [ K * D * D ]
        self.prio_p  = np.ones((self.K, 1)) / self.K        # [ K ]

        for i in range(train_number):
            self.M_step(train_point, self.E_step(train_point))

        return self.mu_set, self.cov_set, self.prio_p

    def test(self, test_point, test_label):
        """
        Test the accuracy using the current EM.

        :param test_point: [ sample_size * D ], np.array
        :param test_label: [ sample_size * K ], np.array
        :return: accuracy, float
        """
        if self.prio_p is None: return 0    # if EM has not been trained

        y = np.argmax(self.E_step(test_point), axis=1)
        t = np.argmax(test_label, axis=1)

        return np.sum(y == t) / len(test_point)
