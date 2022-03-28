import itertools
import numpy as np
import scipy.stats as st


class EM:
    """
    Example:
        em = EM(K)
        em.train(train_point)

        print("EM accuracy: %7.5f"  % em.accuracy(test_point, test_label))
        print("EM precision: %7.5f" % em.precision(test_point, test_label))
        print("EM recall: %7.5f"    % em.recall(test_point, test_label))
    """

    def __init__(self, K):
        self.K = K      # number of clustering / Gaussian mixture
        self.D = None   # dimension, depend on the input point when training

        self.mu_set  = None     # mean of each mixture, [ K * D ]
        self.cov_set = None     # covariance of each mixture, [ K * D * D ]
        self.prio_p  = None     # prior probability of each mixture, [ K ]

    """ Trainer """

    def E_step(self, point):
        """
        Get the posterior probability of input data point
        """
        post_p = np.zeros((len(point), self.K))     # [ N * K ]
        for k in range(self.K):
            post_p[:, k] = self.prio_p[k] * \
                           st.multivariate_normal.pdf(point,
                                                      self.mu_set[k],
                                                      self.cov_set[k],
                                                      allow_singular=True)
        post_p /= np.sum(post_p, axis=1)[:, None]   # [ N * K ]

        return post_p   # posterior probability of input data point, [ N * K ]

    def M_step(self, point, post_p):
        """
        Update prior probability, mu_set, cov_set according to "post_p" from E
        step.
        """
        # sum of posterior probability
        sum_post_p = np.sum(post_p, axis=0)     # [ K ]

        # 1. update prior probability
        self.prio_p = sum_post_p / len(point)   # [ K ]

        for k in range(self.K):
            below = sum_post_p[k]   # float

            # 2. update mu_set
            above = np.sum(point * post_p[:, [k]], axis=0)
            self.mu_set[k] = above / below

            # 3. update cov_set
            x_mu = point - self.mu_set[k]                       # [ N * D ]
            above = np.dot((post_p[:, [k]] * x_mu).T, x_mu)     # [ D * D ]
            self.cov_set[k] = above / below                     # [ D * D ]

        return self.mu_set, self.cov_set, self.prio_p

    def train(self, train_point, epoch=20000, epsilon=1e-10):
        """
        Repeat E step and M step for "epoch" number of iteration.

        Args:
            train_point: [ sample_size * D ], np.array
            epoch: number of iteration, int
            epsilon: stop training when the norm of change of mu < epsilon
        """
        # initialize all parameters
        self.D       = len(train_point[0])                  # int
        self.mu_set  = np.random.randn(self.K, self.D)      # [ K * D ]
        self.cov_set = np.array([np.eye(self.D)] * self.K)  # [ K * D * D ]
        self.prio_p  = np.ones((self.K, 1)) / self.K        # [ K ]

        # train
        for i in range(epoch):
            old_mu = self.mu_set.copy()

            self.M_step(train_point, self.E_step(train_point))

            # breakpoint
            if np.linalg.norm(self.mu_set - old_mu) < epsilon: break

    """ Estimator """

    def order_correction(self, point, label):
        """
        The parameters that EM gets is correct but shuffled since we initialize
        mu of each Gaussian by random. We may get a bad accuracy result
        because the order of the parameters.
            This function will adjust the order of "mu_set," "cov_set," "prio_p"
        that maximize the accuracy.

        Args:
            point: [ sample_size * D ], np.array
            label: [ sample_size * K ], np.array

        Returns:
            accuracy, float
        """
        order = []  # [ K ], store the correct order
        accuracy = 0

        # 1. get the correct order
        t = np.argmax(label, axis=1)
        for j in list(itertools.permutations([i for i in range(self.K)],
                                             self.K)):
            y = np.argmax(self.predict(point)[:, j], axis=1)
            current_accuracy = np.sum(y == t) / len(label)
            if current_accuracy > accuracy:
                order = j
                accuracy = current_accuracy

        # 2. change the order of mu, cov, prior probability
        for data in (self.mu_set, self.cov_set, self.prio_p):
            temp = np.copy(data)    # store the old data
            for i in range(self.K):
                data[i] = temp[order[i]]

        return accuracy

    def predict(self, point):
        return self.E_step(point)

    def accuracy(self, point, label, order_correction=True):
        """
        Return the accuracy
        """
        # 1. check the input data
        if self.prio_p is None:
            return 0  # EM has not been trained
        if len(point[0]) != self.D or len(label[0]) != self.K:
            return 0  # the input label or point is not valid

        # 2. correct the order of parameters
        if order_correction: self.order_correction(point, label)

        # 3. compute the accuracy
        t = np.argmax(label, axis=1)                # actual label
        y = np.argmax(self.predict(point), axis=1)  # predict label
        return np.sum(y == t) / len(label)          # accuracy, float

    def precision(self, point, label, order_correction=True):
        """
        Compute the precision of each cluster and return the average precision.
        """
        # 1. check the input data
        if self.prio_p is None:
            return 0  # EM has not been trained
        if len(point[0]) != self.D or len(label[0]) != self.K:
            return 0  # the input label or point is not valid

        # 2. correct the order of parameters
        if order_correction: self.order_correction(point, label)

        # 3. compute the precision
        t = np.argmax(label, axis=1)                # actual label
        y = np.argmax(self.predict(point), axis=1)  # predict label
        precision = 0
        for k in range(1, self.K):     # find the precision of each cluster
            TP = 0  # Predict class == k, Actual class == k
            FP = 0  # Predict class == k, Actual class != k
            for n in range(len(point)):
                if y[n] != k: continue
                if t[n] == y[n]: TP += 1
                if t[n] != y[n]: FP += 1
            if TP + FP == 0: continue  # avoid divide zero
            precision += TP / (TP + FP)
        return precision / (self.K - 1)

    def recall(self, point, label, order_correction=True):
        """
        Compute the recall of each cluster and return the average recall.
        """
        # 1. check the input data
        if self.prio_p is None:
            return 0  # EM has not been trained
        if len(point[0]) != self.D or len(label[0]) != self.K:
            return 0  # the input label or point is not valid

        # 2. correct the order of parameters
        if order_correction: self.order_correction(point, label)

        # 3. compute the recall
        t = np.argmax(label, axis=1)                # actual label
        y = np.argmax(self.predict(point), axis=1)  # predict label
        recall = 0
        for k in range(1, self.K):     # find the recall of each cluster
            TP = 0  # Predict class == k, Actual class == k
            FN = 0  # Predict class != k, Actual class == k
            for n in range(len(point)):
                if t[n] != k: continue
                if t[n] == y[n]: TP += 1
                if t[n] != y[n]: FN += 1
            if TP + FN == 0: continue  # avoid divide zero
            recall += TP / (TP + FN)
        return recall / (self.K - 1)
