from Bayes import Bayes

import os
import numpy as np
import scipy.stats as st


class EM:
    def __init__(self):
        self.K = None
        self.N = None
        self.D = None

        self.prio_p  = None     # prior probability, [ K ], np.array
        self.mu_set  = None     # [ K * D ], np.array
        self.cov_set = None     # [ K * D * D ], np.array

        self.post_p  = None     # posterior probability, [ N * K ], np.array

    def initialize_EM(self):
        """
        Initialize mu, cov, prior probability, posterior probability.
        Use before start the first E step (before start iteration).
        """
        self.prio_p  = np.ones((self.K, 1)) / self.K        # [ K ]
        self.mu_set  = np.random.randn(self.K, self.D)      # [ K * D ]
        self.cov_set = np.array([np.eye(self.D)] * self.K)  # [ K * D * D ]

        self.post_p  = np.zeros((self.N, self.K))           # [ N * K ]

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
        p = np.zeros((self.N, self.K))
        for k in range(self.K):
            p[:, k] = self.prio_p[k] * \
                      st.multivariate_normal.pdf(sample_point,
                                                 self.mu_set[k],
                                                 self.cov_set[k])
        sumP = np.sum(p, axis=1)
        self.post_p = p / sumP[:, None]     # [ N * K ]

    def M_step(self, sample_point):
        """
        Update prior probability, mu_set, cov_set

        Reference: https://github.com/chenwj1989/MLSP/blob/master/gmm/gmm_em.py

        :param sample_point: [ sample_size * D ], np.array
        """
        sum_post_p = np.sum(self.post_p, axis=0)    # [ K ]

        # update prior probability
        self.prio_p = sum_post_p / self.N   # [ K ]

        for k in range(self.K):
            # update cov_set
            # sum(X * prio_p[k]) -- sum(number * [ N * D ]) -- number
            above = np.sum(np.multiply(sample_point, self.post_p[:, [k]]),
                            axis=0)
            self.mu_set[k] = above / sum_post_p[k]

            # update mu_set
            X_mu = np.subtract(sample_point, self.mu_set[k])   # [ N * D ]
            omega_X_mu_k = np.multiply(self.post_p[:, [k]], X_mu)   # [ N * D ]
            self.cov_set[k] = np.dot(np.transpose(omega_X_mu_k), X_mu) / \
                              sum_post_p[k]

    def train(self, train_point, train_label, test_point, test_label,
              train_number, save_EM=False):
        """
        Repeat E step and M step for "train_number" time.

        :param train_point: [ sample_size * D ], np.array
        :param train_label: [ sample_size * K ], np.array
        :param test_point: [ sample_size * D ], np.array
        :param test_label: [ sample_size * K ], np.array
        :param train_number: number of iteration
        :param save_EM: save "train_accuracy" in file "result" or not, bool
        :return: [ train_number ], np.array
            accuracy for test sample after each train/iteration
        """
        self.K = len(train_label[0])
        self.N, self.D = train_point.shape
        self.initialize_EM()

        test_accuracy = []

        bayes = Bayes(self.mu_set, self.cov_set)    # used to measure accuracy
        for i in range(train_number):
            # train
            self.E_step(train_point)
            self.M_step(train_point)

            # store result
            bayes.mu_set, bayes.cov_set = self.mu_set, self.cov_set
            accuracy = bayes.accuracy(test_point, test_label)
            test_accuracy.append(accuracy)

            # print result
            print("%4d\tA: %7.5f" % (i, accuracy))

        # save result as .csv
        if save_EM:
            if not os.path.exists('result'): os.mkdir('result')
            np.savetxt("result/EM_test_accuracy.csv", test_accuracy,
                       delimiter=",")

        return test_accuracy
