import numpy as np


class Gaussian:
    def __init__(self, N_set, mu_set, cov_set):
        """
        :param N_set: number of sample for each Gaussian, [ K ], np.array
        :param mu_set: mean of each Gaussian, [ K * D ], np.array
        :param cov_set: covariance of each Gaussian, [ K * D * D ], np.array
        """
        # basic dimension parameters
        self.K = len(mu_set)        # number of class, int
        self.D = len(mu_set[0])     # dimension, int
        self.N = sum(N_set)         # total number of points, int
        self.N_k = N_set            # number of points in each class, [ K ]

        # Gaussian parameters
        self.prio_p  = np.divide(self.N_k, self.N)  # [ K ]
        self.mu_set  = mu_set                       # [ K * D ]
        self.cov_set = cov_set                      # [ K * D * D ]

        # sample
        self.point = []     # [ N * D ]
        self.label = []     # [ N * K ]

        # split sample
        self.train_point = []
        self.train_label = []
        self.valid_point = []
        self.valid_label = []
        self.test_point  = []
        self.test_label  = []

        # generate sample and split sample using help function
        self.generate_sample()
        self.split_sample()

    def generate_sample(self):
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
        np.random.shuffle(sample_set)

        self.point = np.array( [x[0] for x in sample_set] )
        self.label = np.array( [x[1] for x in sample_set] )

    def split_sample(self, index_1=0.5, index_2=0.7):
        n_1 = int(index_1 * self.N)
        n_2 = int(index_2 * self.N)
        self.train_point = np.array([self.point[i] for i in range(n_1)])
        self.train_label = np.array([self.label[i] for i in range(n_1)])
        self.valid_point = np.array([self.point[i] for i in range(n_1, n_2)])
        self.valid_label = np.array([self.label[i] for i in range(n_1, n_2)])
        self.test_point = np.array([self.point[i] for i in range(n_2, self.N)])
        self.test_label = np.array([self.label[i] for i in range(n_2, self.N)])
