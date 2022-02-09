import numpy as np
import scipy.stats as st


class EM:
    def __init__(self):
        self.K = None   # number of clustering / Gaussian mixture
        self.D = None   # Dimension

        self.mu_set  = None     # [ K * D ], np.array
        self.cov_set = None     # [ K * D * D ], np.array

        self.prio_p = None      # prior probability, [ K ], np.array
        self.post_p = None      # posterior probability, [ N * K ], np.array

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

        Reference: https://github.com/chenwj1989/MLSP/blob/master/gmm/gmm_em.py

        :param sample_point: [ sample_size * D ], np.array
        """
        p = np.zeros((len(sample_point), self.K))   # [ N * K ]
        for k in range(self.K):
            p[:, k] = self.prio_p[k] * \
                      st.multivariate_normal.pdf(sample_point,
                                                 self.mu_set[k],
                                                 self.cov_set[k])
        sumP = np.sum(p, axis=1)            # [ N ]
        self.post_p = p / sumP[:, None]     # [ N * K ]

    def M_step(self, sample_point):
        """
        Update prior probability, mu_set, cov_set

        Reference: https://github.com/chenwj1989/MLSP/blob/master/gmm/gmm_em.py

        :param sample_point: [ sample_size * D ], np.array
        """
        sum_post_p = np.sum(self.post_p, axis=0)        # [ K ]

        # update prior probability
        self.prio_p = sum_post_p / len(sample_point)    # [ K ]

        for k in range(self.K):
            # update cov_set
            # sum(X * prio_p[k]) -- sum(number * [ N * D ]) -- number
            above = np.sum(np.multiply(sample_point, self.post_p[:, [k]]),
                            axis=0)
            self.mu_set[k] = above / sum_post_p[k]

            # update mu_set
            X_mu = np.subtract(sample_point, self.mu_set[k])        # [ N * D ]
            omega_X_mu_k = np.multiply(self.post_p[:, [k]], X_mu)   # [ N * D ]
            self.cov_set[k] = np.dot(np.transpose(omega_X_mu_k), X_mu) / \
                              sum_post_p[k]

    def predict(self, sample_point):
        """
        By train the EM, we get predict mu, covariance, and prior probability of
        each cluster / Gaussian mixture. Then, we predict the cluster that each
        point of input "sample_point" belongs to. This function will return the
        posterior probability.

        :param sample_point: [ sample_size * D ], np.array
        :return: posterior probability, [ sample_size * K ], np.array
        """
        post_p = np.zeros((len(sample_point), self.K))      # [ N * K ]
        for k in range(self.K):
            post_p[:, k] = self.prio_p[k] * \
                           st.multivariate_normal.pdf(sample_point,
                                                      self.mu_set[k],
                                                      self.cov_set[k])

        return post_p / np.sum(post_p, axis=1)[:, None]   # [ N * K ]

    def train(self, train_point, train_label, train_number):
        """
        Repeat E step and M step for "train_number" time.

        :param train_point: [ sample_size * D ], np.array
        :param train_label: [ sample_size * K ], np.array
        :param train_number: number of iteration
        :return:
        """
        self.K = len(train_label[0])
        self.D = len(train_point[0])

        self.mu_set  = np.random.randn(self.K, self.D)          # [ K * D ]
        self.cov_set = np.array([np.eye(self.D)] * self.K)      # [ K * D * D ]

        self.prio_p = np.ones((self.K, 1)) / self.K             # [ K ]
        self.post_p  = np.zeros((len(train_point), self.K))     # [ N * K ]

        for i in range(train_number):
            self.E_step(train_point)
            self.M_step(train_point)

        return self.mu_set, self.cov_set, self.prio_p

    def test(self, test_point, test_label):
        """
        Test the EM accuracy after train.

        :param test_point: [ sample_size * D ], np.array
        :param test_label: [ sample_size * K ], np.array
        :return: accuracy, float
        """
        y = np.argmax(self.predict(test_point), axis=1)
        t = np.argmax(test_label, axis=1)

        return np.sum(y == t) / test_point.shape[0]
