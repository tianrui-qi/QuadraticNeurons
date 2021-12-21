import numpy as np
import scipy.stats as st


class Bayes:
    def __init__(self, mu_set, cov_set):
        """
        :param mu_set: mean of each Gaussian, [ K * D ]
        :param cov_set: covariance of each Gaussian, [ K * D * D ]
        """
        self.K = len(mu_set)

        self.mu_set  = mu_set
        self.cov_set = cov_set

    def inferences(self, sample_point):
        """
        Bayes Inferences
        Predict the label of the point in the sample point by bayes inferences.
        Notes that the return value is not the true label like [ 0 0 1 0].
        It's the score of each value [ 10 20 30 5 ]. To get the label, still
        need to take argmax.

        :param sample_point: [ sample_size * D ], np.array
        :return: [ sample_size * K ], np.array
        """
        sample_size = len(sample_point)

        bayes_inferences = np.zeros((sample_size, self.K))
        for n in range(sample_size):
            point = sample_point[n]
            for k in range(self.K):
                mu = self.mu_set[k]
                cov = self.cov_set[k]
                probability = st.multivariate_normal.pdf(point, mu, cov)

                bayes_inferences[n][k] = probability

        return bayes_inferences     # [ N * K ]

    def accuracy(self, sample_point, sample_label):
        """
        Give a sample point, get the predicting label from the "predict". Then,
        compare the predicting label with the correct label "sample_label", and
        return the accuracy of the prediction

        :param sample_point: [ sample_size * D ], np.array
        :param sample_label: [ sample_size * K ], np.array
        :return: accuracy of the Bayes prediction (float)
        """
        y = np.argmax(self.inferences(sample_point), axis=1)
        t = np.argmax(sample_label, axis=1)

        return np.sum(y == t) / sample_point.shape[0]
