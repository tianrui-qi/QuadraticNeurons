import random
import numpy as np
import scipy.stats as st


class Gaussian:
    """
    Example:
        # random set parameters
        mu_set = np.array([(np.random.random(D) - 0.5) * 10 for i in range(K)])
        cov_set = []
        for i in range(K):
            a = np.random.random((D, D)) * 2 - 1
            cov = np.dot(a, a.T) + np.dot(a, a.T)
            cov_set.append(cov)
        N_k = [np.random.randint(3000, 6000) for k in range(K)]

        # generate sample
        gaussian = Gaussian(N_k, mu_set, cov_set)
        sample_point, sample_label = train_gaussian.generate_sample()
    """

    def __init__(self, N_k, mu_set, cov_set):
        """
        :param N_k: number of sample for each Gaussian, [ K ], np.array
        :param mu_set: mean of each Gaussian, [ K * D ], np.array
        :param cov_set: covariance of each Gaussian, [ K * D * D ], np.array
        """
        self.K = len(mu_set)    # int
        self.N = sum(N_k)       # int
        self.N_k = N_k          # [ K ]

        self.prio_p  = np.divide(self.N_k, self.N)  # [ K ]
        self.mu_set  = mu_set                       # [ K * D ]
        self.cov_set = cov_set                      # [ K * D * D ]

        self.sample_point = None    # [ N * D ]
        self.sample_label = None    # [ N * K ]

    def generate_sample(self):
        """
        Notes: Use one-hot vector as sample label: for each sample point,
        generate a 1*k matrix that standard for the label of the Gaussian
        Example:
        K = 4
        k = 0: label = [1, 0, 0, 0]
        k = 3: label = [0, 0, 0, 1]

        :return: sample, train and test point and label
            point: [ sample_size * D ], np.array
            label: [ sample_size * K ], np.array
        """
        sample_set = []
        for k in range(self.K):
            # generate N_k[k] number of point for each Gaussian k
            point = np.random.multivariate_normal(self.mu_set[k],
                                                  self.cov_set[k], self.N_k[k])
            # set the label of these point using one-hot vector
            label = np.zeros([self.N_k[k], self.K])
            for n in range(self.N_k[k]):
                label[n][k] = 1
            # append into the sample_set in pair
            for n in range(self.N_k[k]):
                sample_set.append((point[n], label[n]))
        random.shuffle(sample_set)

        self.sample_point = np.array( [x[0] for x in sample_set] )
        self.sample_label = np.array( [x[1] for x in sample_set] )

        return self.sample_point, self.sample_label

    def bayes_inference(self):
        if self.sample_point is None: return

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
