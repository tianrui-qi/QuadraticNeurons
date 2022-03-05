import random
import numpy as np
import scipy.stats as st


class Gaussian:
    def __init__(self, N_k, mu_set, cov_set):
        """
        :param N_k: number of sample for each Gaussian, [ K ], np.array
        :param mu_set: mean of each Gaussian, [ K * D ]
        :param cov_set: covariance of each Gaussian, [ K * D * D ]
        """
        self.K = len(mu_set)    # int
        self.N = sum(N_k)       # int
        self.N_k = N_k          # [ K ]

        self.prio_p  = np.divide(self.N_k, self.N)  # [ K ]
        self.mu_set  = mu_set                       # [ K * D ]
        self.cov_set = cov_set                      # [ K * D * D ]

        self.bg = None

        self.sample_point = None    # [ N * D ]
        self.sample_label = None    # [ N * K ]

    def set_label(self, k, point):
        """
        Generate a 1*k matrix that standard for the label of the Gaussian

        Example:
        label = 0, k = 4
            self.label = [1, 0, 0, 0]
        label = 3, k = 4
            self.label = [0, 0, 0, 1]

        :param k: label index of this Gaussian
        :param point: the point we want to label
        :return: a 1*k matrix that standard for the label of the Gaussian
        """
        if self.bg:
            if len(self.N_k) == self.K: self.N_k = np.append(self.N_k, 0)

            sample_label = np.zeros([len(point), self.K+1])
            probability = st.multivariate_normal.pdf(
                point, self.mu_set[k], self.cov_set[k])
            for n in range(len(point)):
                if probability[n] < 0.0455:     # 0.0027, 0.0455
                    sample_label[n][self.K] = 1
                    self.N_k[self.K] += 1
                    self.N_k[k] -= 1
                else:
                    sample_label[n][k] = 1
        else:
            sample_label = np.zeros([len(point), self.K])
            for n in range(len(point)):
                sample_label[n][k] = 1

        return sample_label

    def generate_sample(self, bg=False):
        """
        :param bg: add a new cluster "Background" (2-sigma) or not, use when
                    set label of each point.
        :return: sample, train and test point and label
            point: [ sample_size * D ], np.array
            label: [ sample_size * K ], np.array
        """
        self.bg = bg

        sample_set = []

        for k in range(self.K):
            point = np.random.multivariate_normal(self.mu_set[k],
                                                  self.cov_set[k], self.N_k[k])
            label = self.set_label(k, point)
            for n in range(self.N_k[k]):
                sample_set.append((point[n], label[n]))
        random.shuffle(sample_set)

        self.sample_point = np.array( [x[0] for x in sample_set] )
        self.sample_label = np.array( [x[1] for x in sample_set] )

        return self.sample_point, self.sample_label

    def bayes_inference(self):
        if self.sample_point is None or self.bg is None: return

        post_p = np.zeros((self.N, self.K))     # [ N * K ]
        for k in range(self.K):
            post_p[:, k] = self.prio_p[k] * \
                           st.multivariate_normal.pdf(self.sample_point,
                                                      self.mu_set[k],
                                                      self.cov_set[k],
                                                      allow_singular=True)
        t = np.argmax(self.sample_label, axis=1)
        y = np.argmax(post_p, axis=1)

        return np.sum(y == t) / self.N
