import os
import random
import numpy as np


class Gaussian:
    def __init__(self, mu_set, cov_set):
        """
        :param mu_set: mean of each Gaussian, [ K * D ]
        :param cov_set: covariance of each Gaussian, [ K * D * D ]
        """
        self.K = len(mu_set)

        self.mu_set  = mu_set
        self.cov_set = cov_set

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

    def set_label(self, k):
        """
        Generate a 1*k matrix that standard for the label of the Gaussian

        Example:
        label = 0, k = 4
            self.label = [1, 0, 0, 0]
        label = 3, k = 4
            self.label = [0, 0, 0, 1]

        :param k: label index of this Gaussian
        :return: a 1*k matrix that standard for the label of the Gaussian
        """
        sample_label = np.zeros(self.K)
        sample_label[k] = 1
        return sample_label

    def save_sample(self):
        """
        Save all the point and label if save_sample=True
        """
        if not os.path.exists('sample'): os.mkdir('sample')
        np.savetxt("sample/sample_point.csv", self.sample_point, delimiter=",")
        np.savetxt("sample/sample_label.csv", self.sample_label, delimiter=",")
        np.savetxt("sample/train_point.csv",  self.train_point,  delimiter=",")
        np.savetxt("sample/train_label.csv",  self.train_label,  delimiter=",")
        np.savetxt("sample/test_point.csv",   self.test_point,   delimiter=",")
        np.savetxt("sample/test_label.csv",   self.test_label,   delimiter=",")

    def load_sample(self):
        """
        Load all the point and label if load_sample=True
        """
        self.sample_point = np.loadtxt("sample/sample_point.csv", delimiter=",")
        self.sample_label = np.loadtxt("sample/sample_label.csv", delimiter=",")
        self.train_point  = np.loadtxt("sample/train_point.csv",  delimiter=",")
        self.train_label  = np.loadtxt("sample/train_label.csv",  delimiter=",")
        self.test_point   = np.loadtxt("sample/test_point.csv",   delimiter=",")
        self.test_label   = np.loadtxt("sample/test_label.csv",   delimiter=",")

    def generate_sample(self, N_k, load_sample=False, save_sample=False):
        """
        If sample already been created ("sample" file exist), the function will
        load from the file directly. If not, it will creat sample point and
        corresponding label for each Gaussian, shuffle (random) them at the
        same time, and split the sample point into two part: train and test.
        Then store them in the file "sample" (if "save_sample"=True).

        :param N_k: number of sample for each Gaussian
        :param load_sample: load sample from file "sample" or creat new
        :param save_sample: save sample in file "sample" or not
        :return: sample, train and test point and label
            point: [ sample_size * D ], np.array
            label: [ sample_size * K ], np.array
        """
        # 1. get sample point and label

        sample_set = []

        for k in range(self.K):
            point = self.set_point(k, N_k)
            label = self.set_label(k)
            for n in range(N_k):
                sample_set.append((point[n], label))
        random.shuffle(sample_set)

        self.sample_point = np.array( [x[0] for x in sample_set] )
        self.sample_label = np.array( [x[1] for x in sample_set] )

        # 2. split point and label into train and test

        train_point, train_label = [], []
        test_point, test_label = [], []

        N = len(self.sample_point)  # N = K * N_k
        N_test = int(N / 3)
        for i in range(N_test):
            test_point.append(self.sample_point[i])
            test_label.append(self.sample_label[i])
        for i in range(N_test, N):
            train_point.append(self.sample_point[i])
            train_label.append(self.sample_label[i])

        self.train_point = np.array(train_point)
        self.train_label = np.array(train_label)
        self.test_point = np.array(test_point)
        self.test_label = np.array(test_label)

        # 3. load and save

        if load_sample and os.path.exists('sample'): self.load_sample()
        if save_sample: self.save_sample()

        return self.train_point, self.train_label, \
               self.test_point, self.test_label, \
               self.sample_point, self.sample_label
