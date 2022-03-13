import time
import itertools
import numpy as np
import scipy.stats as st


class EM:
    """
    With the given number of clusters, K, do the EM process with train point to
    get each Gaussian's parameters including mean, covariance, and prior
    probability.

    Example 1:
        em = EM(K)
        em.train(train_point)
        print("EM accuracy: %7.5f" % em.test(test_point, test_label))
    Example 2:
        em = EM(K)
        em.train(train_point, test_point=test_point, test_label=test_label)
        print("EM test accuracy: %7.5f" % em.test_accuracy)

    The result including time and accuracy will be store in self.train_time,
    self.test_time, self.train_accuracy, self.valid_accuracy, and
    self.test_accuracy.
    """

    def __init__(self, K):
        """
        :param K: number of clustering / Gaussian mixture, int
        """
        self.K = K
        self.D = None

        self.mu_set  = None     # [ K * D ]
        self.cov_set = None     # [ K * D * D ]
        self.prio_p  = None     # [ K ]

        # result
        self.train_time = []
        self.test_time  = []
        self.train_accuracy = []
        self.valid_accuracy = []
        self.test_accuracy  = []

    def E_step(self, point):
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
        By train the EM, we get predict mu, covariance, and prior probability of
        each cluster / Gaussian mixture. Then, we predict the cluster that each
        point of input "sample_point" belongs to. This function will return the
        posterior probability.

        :param point: [ sample_size * D ], np.array
        :return: posterior probability, [ sample_size * K ], np.array
        """
        post_p = np.zeros((len(point), self.K))  # [ N * K ]
        for k in range(self.K):
            post_p[:, k] = self.prio_p[k] * \
                           st.multivariate_normal.pdf(point,
                                                      self.mu_set[k],
                                                      self.cov_set[k],
                                                      allow_singular=True)

        return post_p/np.sum(post_p, axis=1)[:, None]   # [ N * K ]

    def M_step(self, point, post_p):
        """
        Update prior probability, mu_set, cov_set according to "post_p"

        :param point: [ sample_size * D ], np.array
        :param post_p: posterior probability, [ N * K ], np.array
        :return self.mu_set: [ K * D ], np.array
        :return self.cov_set: [ K * D * D ], np.array
        :return self.prio_p: prior probability, [ K ], np.array
        """
        sum_post_p = np.sum(post_p, axis=0)             # [ K ]

        # update prior probability
        self.prio_p = sum_post_p / len(point)    # [ K ]

        for k in range(self.K):
            below = sum_post_p[k]   # float

            # update mu_set
            above = np.sum(point * post_p[:, [k]], axis=0)
            self.mu_set[k] = above / below

            # update cov_set
            x_mu = point - self.mu_set[k]                # [ N * D ]
            above = np.dot((post_p[:, [k]] * x_mu).T, x_mu)     # [ D * D ]
            self.cov_set[k] = above / below                     # [ D * D ]

        return self.mu_set, self.cov_set, self.prio_p

    def predict(self, point):
        return self.E_step(point)

    def order_correction(self, point, label):
        """
        The parameters that EM gets is correct but shuffled since we initialize
        mu of each Gaussian by random. We way got a bad accuracy result
        because the order of the parameters.
            This function will adjust the order of "mu_set," "cov_set," "prio_p"
        that maximize the accuracy.

        :param point: [ sample_size * D ], np.array
        :param label: [ sample_size * K ], np.array
        :return: accuracy, float
        """
        order = []  # [ K ], store the correct order
        accuracy = 0

        # get the correct order
        t = np.argmax(label, axis=1)
        for j in list(itertools.permutations([i for i in range(self.K)],
                                             self.K)):
            y = np.argmax(self.E_step(point)[:, j], axis=1)
            current_accuracy = np.sum(y == t) / len(label)
            if current_accuracy > accuracy:
                order = j
                accuracy = current_accuracy

        # change the order of mu, cov, prior probability according to the order
        for data in (self.mu_set, self.cov_set, self.prio_p):
            temp = np.copy(data)    # store the old data
            for i in range(self.K):
                data[i] = temp[order[i]]

        return accuracy

    def test(self, point, label, order_correction=True):
        """
        Test the accuracy using the current EM.

        :param point: [ sample_size * D ], np.array
        :param label: [ sample_size * K ], np.array
        :param order_correction: call the help function "self.order_correction"
            to correct the order of parameters or not. You can also call the
            function before test.
        :return: accuracy, float
        """
        if self.prio_p is None: return 0    # means EM has not been trained
        if order_correction: self.order_correction(point, label)

        t = np.argmax(label, axis=1)
        y = np.argmax(self.E_step(point), axis=1)
        return np.sum(y == t) / len(label)

    def train(self, train_point, train_label=None,
              valid_point=None, valid_label=None,
              test_point=None, test_label=None,
              epoch=20000, stop_point=50):
        """
        Repeat E step and M step for "epoch" number of iteration.

        :param train_point: [ sample_size * D ], np.array
        :param train_label: [ sample_size * K ], np.array
        :param valid_point: [ sample_size * D ], np.array
        :param valid_label: [ sample_size * K ], np.array
        :param test_point: [ sample_size * D ], np.array
        :param test_label: [ sample_size * K ], np.array
        :param epoch: number of iteration, int
        :param stop_point: stop training after "stop_point" number of
            iteration such that the accuracy of validation set does not increase
        :return self.mu_set: [ K * D ], np.array
        :return self.cov_set: [ K * D * D ], np.array
        :return self.prio_p: prior probability, [ K ], np.array
        """
        # initialize all parameters
        self.D = len(train_point[0])
        self.mu_set  = np.random.randn(self.K, self.D)      # [ K * D ]
        self.cov_set = np.array([np.eye(self.D)] * self.K)  # [ K * D * D ]
        self.prio_p  = np.ones((self.K, 1)) / self.K        # [ K ]

        # variable use to store result including time and accuracy
        train_time = np.zeros([epoch])
        test_time = []

        train_accuracy = np.zeros([epoch])
        valid_accuracy = np.zeros([epoch])
        test_accuracy = []

        # train
        time_track = 0
        stop_track = 0
        accuracy_track = 0
        for i in range(epoch):
            begin = time.time()

            # Main part ========================================================
            self.M_step(train_point, self.E_step(train_point))
            # ==================================================================

            time_track += time.time() - begin
            train_time[i] = time_track

            """
            if train_label is not None:
                train_accuracy[i] = self.test(train_point, train_label)
            """
            if valid_label is not None:
                valid_accuracy[i] = self.test(valid_point, valid_label)

            # Early Stopping ===================================================
            if valid_label is None:
                continue
            elif stop_point < stop_track:
                break
            elif valid_accuracy[i] > accuracy_track:
                stop_track = 0
                accuracy_track = valid_accuracy[i]
            else:
                stop_track += 1
            # ==================================================================

        # test
        if test_label is not None:
            begin = time.time()
            test_accuracy.append(self.test(test_point, test_label,
                                           order_correction=False))
            test_time.append(time.time() - begin)

        # store
        self.train_time = train_time
        self.test_time = test_time

        #### self.train_accuracy = train_accuracy
        #### self.valid_accuracy = valid_accuracy
        self.test_accuracy = test_accuracy

        return self.mu_set, self.cov_set, self.prio_p
