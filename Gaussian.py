import random
import numpy as np
import scipy.stats as st


class Gaussian:
    def __init__(self, mu_set, cov_set):
        """
        :param mu_set: mean of each Gaussian, [ K * D ]
        :param cov_set: covariance of each Gaussian, [ K * D * D ]
        """
        self.K = len(mu_set)

        self.mu_set  = mu_set
        self.cov_set = cov_set
        self.N_k = None

        self.bg = None

        self.sample_point = None
        self.sample_label = None
        self.train_point  = None
        self.train_label  = None
        self.test_point   = None
        self.test_label   = None

    def set_point(self, k, N_k):
        """
        Generate n number of sample point according to the Gaussian Distribution

        :param k: index of the Gaussian
        :param N_k: number of sample point that need to generate
        :return: the list of sample point that generate for Gaussian k
        """
        return np.random.multivariate_normal(self.mu_set[k], self.cov_set[k],
                                             N_k)

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

    def generate_sample(self, N_k, bg=False):
        """
        If sample already been created ("sample" file exist), the function will
        load from the file directly. If not, it will creat sample point and
        corresponding label for each Gaussian, shuffle (random) them at the
        same time, and split the sample point into two part: train and test.
        Then store them in the file "sample" (if "save_sample"=True).

        :param N_k: number of sample for each Gaussian, [ K ], np.array
        :param bg: add a new cluster "Background" (2-sigma) or not, use when
                    set label of each point.
        :return: sample, train and test point and label
            point: [ sample_size * D ], np.array
            label: [ sample_size * K ], np.array
        """
        self.N_k = N_k
        self.bg = bg

        # 1. get sample point and label

        sample_set = []

        for k in range(self.K):
            point = self.set_point(k, self.N_k[k])
            label = self.set_label(k, point)
            for n in range(self.N_k[k]):
                sample_set.append((point[n], label[n]))
        random.shuffle(sample_set)

        self.sample_point = np.array( [x[0] for x in sample_set] )
        self.sample_label = np.array( [x[1] for x in sample_set] )

        # 2. split point and label into train and test

        N = len(self.sample_point)  # N = K * N_k
        N_test = int(N / 3)
        self.test_point = np.array([self.sample_point[i] for i in range(N_test)])
        self.test_label = np.array([self.sample_label[i] for i in range(N_test)])

        self.train_point = np.array([self.sample_point[i]
                                     for i in range(N_test, N)])
        self.train_label = np.array([self.sample_label[i]
                                     for i in range(N_test, N)])

        return self.train_point, self.train_label, \
               self.test_point, self.test_label, \
               self.sample_point, self.sample_label
