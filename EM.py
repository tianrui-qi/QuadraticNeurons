import numpy as np
import scipy.stats as st
from Bayes import Bayes

class EM:
    def __init__(self, sample_point, sample_label):
        self.K = len(sample_label[0])
        self.N, self.D = sample_point.shape

        self.prio_p  = np.ones((self.K,1)) / self.K         # [ K ]
        self.mu_set  = np.random.randn(self.K, self.D)      # [ K * D ]
        self.cov_set = np.array([np.eye(self.D)] * self.K)  # [ K * D * D ]

        self.post_p = np.zeros((self.N, self.K))  # [ N * K ]

        self.sample_point = sample_point    # [ N * D ]
        self.sample_label = sample_label    # [ N * K ]

    def E_step(self):
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
        """
        p = np.zeros((self.N, self.K))
        for k in range(self.K):
            p[:, k] = self.prio_p[k] * \
                      st.multivariate_normal.pdf(self.sample_point,
                                                 self.mu_set[k],
                                                 self.cov_set[k])
        sumP = np.sum(p, axis=1)
        self.post_p = p / sumP[:, None]     # [ N * K ]

    def M_step(self):
        """
        Update prior probability, mu_set, cov_set

        Reference: https://github.com/chenwj1989/MLSP/blob/master/gmm/gmm_em.py
        """
        sum_post_p = np.sum(self.post_p, axis=0)    # [ K ]

        # update prior probability
        self.prio_p = sum_post_p / self.N   # [ K ]

        for k in range(self.K):
            # update cov_set
            # sum(X * prio_p[k]) -- sum(number * [ N * D ]) -- number
            above = np.sum(np.multiply(self.sample_point, self.post_p[:, [k]]),
                            axis=0)
            self.mu_set[k] = above / sum_post_p[k]

            # update mu_set
            X_mu = np.subtract(self.sample_point, self.mu_set[k])   # [ N * D ]
            omega_X_mu_k = np.multiply(self.post_p[:, [k]], X_mu)   # [ N * D ]
            self.cov_set[k] = np.dot(np.transpose(omega_X_mu_k), X_mu) / \
                              sum_post_p[k]

    def train(self, train_number):
        bayes = Bayes(self.mu_set, self.cov_set)
        for i in range(train_number):
            self.E_step()
            self.M_step()
            bayes.mu_set, bayes.cov_set = self.mu_set, self.cov_set
            bayes_accuracy = bayes.bayes_accuracy(self.sample_point,
                                                  self.sample_label)
            print("%4d\tEM Accuracy: %10.7f" % (i, bayes_accuracy))
            if i % 5 == 0:
                bayes.plot_decision_boundary(self.sample_point,
                                             plot_confidence_interval=True)
