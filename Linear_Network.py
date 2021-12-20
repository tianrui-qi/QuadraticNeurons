import os
import numpy as np
import matplotlib.patches as mp
import matplotlib.pyplot as plt


class Linear_Network:
    def __init__(self, dimension, neuron_num, activation_func,
                 load_network=False):
        # basic dimension parameter
        self.L = len(neuron_num)         # number of hidden & output layer
        self.D = dimension               # dimension of sample data point
        self.K = neuron_num[self.L - 1]  # number of Gaussian / classifications

        # network parameter
        self.neuron_num      = neuron_num
        self.activation_func = activation_func
        self.para            = {}

        # optimizer parameter
        self.h = {}     # for optimizer "AdaGrad", "RMSprop"
        self.m = {}     # for optimizer "Adam"
        self.v = {}     # for optimizer "Adam"

        self._initialize_network(load_network)

        # result
        self.train_loss_set = []
        self.test_loss_set  = []
        self.train_accuracy_set  = []
        self.test_accuracy_set   = []

    """ Constructor """

    def _initialize_network(self, load_network):
        """
        Initialize five dictionary of the parameters of object "Linear_Network."

        See https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf (English)
            https://arxiv.org/abs/1502.01852 (English)
        about the initialization of parameter weighs and bias.

        :param load_network: bool
            If the parameters load from file "Linear_Network." Notes that the
            network are saved by the help function "save_network"
        """
        if len(self.activation_func) != self.L:
            print('Error! Dimension of the "activation_func" not match!')

        for l in range(self.L):
            if l == 0:
                node_from = self.D
            else:
                node_from = self.neuron_num[l - 1]
            node_to   = self.neuron_num[l]

            # network parameter
            sd = 0.01
            if self.activation_func[l] == self.sigmoid:
                sd = np.sqrt(1 / node_from)
            elif self.activation_func[l] == self.relu:
                sd = np.sqrt(2 / node_from)
            self.para['w'+str(l)] = sd * np.random.randn(node_from, node_to)
            self.para['b'+str(l)] = np.zeros( (1, node_to) )

            # optimizer index
            self.h['w'+str(l)] = np.zeros( (node_from, node_to) )
            self.h['b'+str(l)] = np.zeros( (        1, node_to) )
            self.m['w'+str(l)] = np.zeros( (node_from, node_to) )
            self.m['b'+str(l)] = np.zeros( (        1, node_to) )
            self.v['w'+str(l)] = np.zeros( (node_from, node_to) )
            self.v['b'+str(l)] = np.zeros( (        1, node_to) )

        if load_network: self.load_network()

    """ Data Processing """

    @staticmethod
    def normalize(sample_point, min_val=-1, max_val=1):
        """
        Adjusting values measured on different scales to a notionally common
        scale ("min_val" - "max_val" for this case). Notes that the distribution
        of the point do not change.

        :param sample_point: [ sample_size * D ], np.array
        :param min_val: minimum of the sample point after normalize, int
        :param max_val: maximum of the sample point after normalize, int
        :return: the sample point after normalize, [ sample_size * D ], np.array
        """
        min_x = np.min(sample_point)
        max_x = np.max(sample_point)
        scale = float(max_val - min_val) / (max_x - min_x)
        shift = float((max_val + min_val) - (max_x + min_x)) / 2

        sample_point = (sample_point + shift) * scale

        return sample_point     # [ sample_size * D ], np.array

    @staticmethod
    def split_data(sample_point, sample_label):
        """
        Split the sample point into two part: train and test.
        Number of train = (1-1/K) * N
        Number of test  = (  1/K) * N = N - Number of train

        :param sample_point: [ sample_size * D ], np.array
        :param sample_label: [ sample_size * K ], np.array
        :return: train and test point and label that we will use in "train"
        """
        N = len(sample_point)
        N_test = int(N/len(sample_label[0]))

        train_point, train_label = [], []
        test_point, test_label = [], []

        for i in range(N_test):
            test_point.append(sample_point[i])
            test_label.append(sample_label[i])
        for i in range(N_test, N):
            train_point.append(sample_point[i])
            train_label.append(sample_label[i])

        return np.array(train_point), np.array(train_label), \
               np.array(test_point), np.array(test_label)

    # noinspection PyTypeChecker
    def save_network(self):
        """
        Save all the parameters of the network and result in the file
        "Linear_Network." Notes that the network will be saved only when
        variable of "train", "save_network" is True.
        """
        if not os.path.exists('Linear_Network'): os.mkdir('Linear_Network')
        np.savetxt("Linear_Network/train_loss.txt", self.train_loss_set)
        np.savetxt("Linear_Network/train_accuracy.txt", self.train_accuracy_set)
        np.savetxt("Linear_Network/test_loss.txt", self.test_loss_set)
        np.savetxt("Linear_Network/test_accuracy.txt", self.test_accuracy_set)

        for key in self.para.keys():
            np.savetxt("Linear_Network/para_{}.txt".format(key), self.para[key])
        for key in self.h.keys():
            np.savetxt("Linear_Network/h_{}.txt".format(key), self.h[key])
        for key in self.m.keys():
            np.savetxt("Linear_Network/m_{}.txt".format(key), self.m[key])
        for key in self.v.keys():
            np.savetxt("Linear_Network/v_{}.txt".format(key), self.v[key])

    def load_network(self):
        """
        Load all the parameters of the network from the file "Linear_Network".
        Notes that the network's parameters will be initialized by the help
        function only when variable of "_initialize_network", "load_network" is
        True.
        """
        if not os.path.exists('Linear_Network'): return
        for l in range(self.L):
            for k in ('w', 'b'):
                self.para[k + str(l)] = \
                    np.loadtxt("Linear_Network/para_{}.txt".format(k + str(l)))
                self.h[k + str(l)] = \
                    np.loadtxt("Linear_Network/h_{}.txt".format(k + str(l)))
                self.m[k + str(l)] = \
                    np.loadtxt("Linear_Network/m_{}.txt".format(k + str(l)))
                self.v[k + str(l)] = \
                    np.loadtxt("Linear_Network/v_{}.txt".format(k + str(l)))

    """ Estimator """

    def predict(self, sample_point):
        """
        Take "sample_point" as the network's input. Using the current network
        parameters to predict the label of the "sample_point." Notes that the
        return value is not the true label like [ 0 0 1 0]. It's the score of
        each value [ 10 20 30 5 ]. To get the label, still need to take argmax
        of return value "z"

        :param sample_point:  [ sample_size * D ], np.array
        :return: [ sample_size * K ], np.array
        """
        a = sample_point
        for l in range(self.L):
            z = np.dot(a, self.para['w'+str(l)]) + self.para['b'+str(l)]
            a = self.activation_func[l](z)

        return a

    def accuracy(self, sample_point, sample_label):
        """
        Give a sample point, get the predicting label from the network. Then,
        compare the predicting label with the correct label "sample_label", and
        return the accuracy of the prediction

        correct = 0
        for n in range(sample_size):
            if y[n] == t[n]:
                correct = correct + 1
        accuracy = correct / sample_size

        :param sample_point: [ sample_size * D ], np.array
        :param sample_label: [ sample_size * K ], np.array
        :return: accuracy of the network prediction (float)
        """
        y = np.argmax(self.predict(sample_point), axis=1)
        t = np.argmax(sample_label, axis=1)

        return np.sum(y == t) / sample_point.shape[0]

    """ Three Activation Functions """

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        if x.ndim == 2:
            x = x.T
            x -= np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    """ One Loss Function """

    def CRE(self, sample_point, sample_label):
        """
         "Cross Entropy Error"

         The t is correct label. The first layer of for loop is using to traverse
         entire sample, and the second layer of for loop is using to control
         which y value of that sample will affect the loss value. The "y" value
         will affect the loss value only when it's in the same position of
         correct label.

         Example,
                 t_i = [  0  0  1  0 ]       y_i = [ 10 20 10  2 ]
         then, for sample point i,
         loss_i = ( 0 * 10 ) + ( 0 * 20 ) + ( 1 * 10 ) + ( 0 * 2 ) = 1 * 10 = 10

         :param sample_point: [ sample_size * D ], np.array
         :param sample_label: [ sample_size * K ], np.array
         :return: loss value (float)
         """
        y = self.predict(sample_point)
        t = sample_label

        delta = 1e-7
        return -(np.sum(np.multiply(t, np.log(y + delta))) /
                 sample_point.shape[0])

    """ Two Gradient Calculator """

    def gradient_ng(self, sample_point, sample_label):
        """
        "Numerical Gradient"

        Get the gradient of all the parameter by just forward, no backward.
        The first loop is used to go through all the parameter.
        Inside the first loop is the numerical gradient of that parameter:
        give a small change 'h', see how f, the loss function, change.

        :param sample_point: [ sample_size * D ], np.array
        :param sample_label: [ sample_size * K ], np.array
        :return: dictionary, gradient for all the parameter
        """
        grad = {}
        for key in self.para.keys():
            h = 1e-4  # 0.0001

            grad[key] = np.zeros_like(self.para[key])

            it = np.nditer(self.para[key],
                           flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                tmp_val = self.para[key][idx]

                self.para[key][idx] = float(tmp_val) + h
                fxh1 = self.CRE(sample_point, sample_label)  # f(x+h)

                self.para[key][idx] = float(tmp_val) - h
                fxh2 = self.CRE(sample_point, sample_label)  # f(x-h)

                grad[key][idx] = (fxh1 - fxh2) / (2 * h)

                self.para[key][idx] = tmp_val
                it.iternext()

        return grad

    def gradient_bp(self, sample_point, sample_label):
        """
        "Backpropagation"

        Use backpropagation to find every parameters' gradient with less time
        complexity than Numerical Gradient "gradient_ng".

        We first take forward step, which same to "predict", the only different
        is we record the "a" value using a dictionary since we need to use it
        during backward step.

        For each layer, we have three step:
             a_i  ---[w_i,b_i]--->  z_i  ---[activation_func]--->  a_(i+1)
        a_i: the input of the current layer, which is also the output of
             previous layer's z_(i-1).
        z_i: a_i * w_i + b_i
        a_(i+1): activation_function (z_i)

        In backward step, we just reverse all the step above
          da_i  <---[dw_i,db_i]---  dz_i  <---[d_activation_func]---  da_(i+1)
        The only difference is that the things we got now is "d", gradient.

        :param sample_point: [ sample_size * D ], np.array
        :param sample_label: [ sample_size * K ], np.array
        :return: dictionary, gradient for all the parameter
        """
        # forward
        # a0 -> w0,b0 -> z0 -> a1 -> w1,b1 -> z1 -> a2
        a = {0: sample_point}
        for l in range(self.L):
            z = np.dot(a[l], self.para['w' + str(l)]) + self.para['b' + str(l)]
            a[l + 1] = self.activation_func[l](z)

        # backward
        # da0 <- dw0,db0 <- dz0 <- da1 <- dw1,db1 <- dz1 <- da2
        dw = {}
        db = {}

        da = 0
        for l in range(self.L-1, -1, -1):
            if l == self.L-1:
                dz = (a[l + 1] - sample_label) / len(sample_point)
            else:
                dz = da * (1.0 - a[l + 1]) * a[l + 1]
            da = np.dot(dz, self.para['w' + str(l)].T)
            dw[l] = np.dot(a[l].T, dz)
            db[l] = np.sum(dz, axis=0)

        # get gradient
        grad = {}
        for l in range(self.L):
            grad['w'+str(l)] = dw[l]
            grad['b'+str(l)] = db[l]

        return grad

    """ Four Optimizers """

    def SGD(self, grad, para):
        """
        "Stochastic Gradient Descent"

        Update all parameters including weighs and biases

        :param grad: dictionary
        :param para: dictionary, parameter that need for all optimizer
            ( will use "lr" )
        """
        for key in grad.keys():
            self.para[key] -= para["lr"] * grad[key]

    def AdaGrad(self, grad, para):
        """
        "Adaptive Gradient Algorithm", an improvement basis on "SGD" above

        Update all parameters including weighs and biases
        Can adjust learning rate by h
        At beginning, since the sum of squares of historical gradients is
        smaller, h += grads * grads the learning rate is high at first.
        As the sum of squares of historical gradients become larger, the
        learning rate will decrease

        :param grad:dictionary
        :param para: dictionary, parameter that need for all optimizer
            ( will use "lr" )
        """
        delta = 1e-7  # avoid divide zero
        for key in grad.keys():
            self.h[key] += np.square(grad[key] )
            self.para[key] -= para["lr"] * grad[key] / \
                              (np.sqrt(self.h[key]) + delta)

    def RMSprop(self, grad, para):
        """
        "Root Mean Squared Propagation", an improvement basis on "AdaGrad" above

        See "https://zhuanlan.zhihu.com/p/34230849" (Chinese)

        Update all parameters including weighs and biases
        Use "decay_rate" to control how much historical information (h) is
        retrieved.
        When the sum of squares of historical gradients is smaller,
        which means that the parameters space is gentle, (h) will give a larger
        number, which will increase the learning rate.
        When the sum of squares of historical gradients is larger, which means
        that the parameters space is steep, (h) will give a smaller number,
        which will decrease the learning rate.

        :param grad: dictionary
        :param para: dictionary, parameter that need for all optimizer
            ( will use "lr", "decay_rate" )
        """
        delta = 1e-7  # avoid divide zero
        for key in grad.keys():
            self.h[key] *= para["decay_rate"]
            self.h[key] += (1.0 - para["decay_rate"]) * np.square(grad[key])
            self.para[key] -= para["lr"] * grad[key] / \
                              (np.sqrt(self.h[key]) + delta)

    def Adam(self, grad, para):
        """
        "Adaptive Moment Estimation", an improvement basis on "RMSprop" above
        and momentum

        See "https://arxiv.org/abs/1412.6980v8" (English)

        :param grad: dictionary
        :param para: dictionary, parameter that need for all optimizer
            ( will use "lr", "beta1", "beta2", "iter" )
        """
        para["iter"] += 1
        lr_t = para["lr"] * np.sqrt(1.0 - para["beta2"]**para["iter"]) / \
               (1.0 - para["beta1"]**para["iter"])
        delta = 1e-7  # avoid divide zero
        for key in grad.keys():
            self.m[key] += (1.0 - para["beta1"]) * (grad[key] - self.m[key])
            self.v[key] += (1.0 - para["beta2"]) * (grad[key]**2 - self.v[key])
            self.para[key] -= lr_t*self.m[key] / (np.sqrt(self.v[key]) + delta)

    """ Trainer """

    def train(self, sample_point, sample_label, test_point, test_label,
              train_number, gradient, optimizer, optimizer_para,
              save_network=False):
        # data processing
        sample_point = self.normalize(sample_point)
        test_point = self.normalize(test_point)

        for i in range(train_number):
            # train
            grad = gradient(self, sample_point, sample_label)
            optimizer(self, grad, optimizer_para)

            # store result
            train_loss     = self.CRE(sample_point, sample_label)
            train_accuracy = self.accuracy(sample_point, sample_label)
            self.train_loss_set.append(train_loss)
            self.train_accuracy_set.append(train_accuracy)

            test_loss     = self.CRE(test_point, test_label)
            test_accuracy = self.accuracy(test_point, test_label)
            self.test_loss_set.append(test_loss)
            self.test_accuracy_set.append(test_accuracy)

            # print result
            print('%4d\tL: %10.7f\tA: %7.5f\tL: %10.7f\tA: %7.5f' %
                  (i, train_loss, train_accuracy, test_loss, test_accuracy))
        if save_network: self.save_network()

    """ Visualization """

    def plot_result(self, bayes_accuracy):
        # plot the accuracy cure
        plt.plot(bayes_accuracy + np.zeros_like(self.train_accuracy_set),
                 color="blue")
        plt.plot(self.train_accuracy_set, color="red")
        plt.plot(self.test_accuracy_set, color="green")

        plt.legend(["Bayes", "Linear NN (train)", "Linear NN (test)"],
                   fontsize=14)
        plt.title("Linear Neural Network Accuracy (Detail)")
        plt.xlabel("Train Number")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.grid()
        plt.show()

        # plot the detail accuracy cure
        plt.plot(bayes_accuracy + np.zeros_like(self.train_accuracy_set),
                 linewidth=2, color="blue")
        plt.plot(self.train_accuracy_set, color="red")
        plt.plot(self.test_accuracy_set, color="green")

        plt.legend(["Bayes", "Linear NN (train)", "Linear NN (test)"],
                   fontsize=14)
        plt.title("Linear Neural Network Accuracy (Detail)")
        plt.xlabel("Train Number")
        plt.ylabel("Accuracy")
        plt.ylim(bayes_accuracy-0.015, bayes_accuracy+0.005)
        plt.grid()
        plt.show()

    def plot_confidence_interval(self, mu_set, cov_set, ax, color):
        """
        Help function for "plot_decision_boundary", using to plot the confident
        interval ellipse (99.73%) of the normal distribution

        :param mu_set: mean set, mean of each Gaussian, [ K * ... ]
        :param cov_set: covariance of each Gaussian, [ K * ... ]
        :param ax: axes object of the 'fig'
        :param color: color of the ellipse
        """
        # P Value of Chi-Square->99.73%: 11.8 ; 95.45%: 6.18 ; 68.27%: 2.295
        # P Value from Chi-Square Calculator:
        # https://www.socscistatistics.com/pvalues/chidistribution.aspx
        confidence = 11.8
        for k in range(self.K):
            # calculate eigenvalue and eigenvector
            eigenvalue, eigenvector = np.linalg.eig(cov_set[k])
            sqrt_eigenvalue = np.sqrt(np.abs(eigenvalue))

            # calculate all the parameter needed for plotting ellipse
            width = 2 * np.sqrt(confidence) * sqrt_eigenvalue[0]
            height = 2 * np.sqrt(confidence) * sqrt_eigenvalue[1]
            angle = np.rad2deg(np.arccos(eigenvector[0, 0]))

            # plot the ellipse
            ell = mp.Ellipse(xy=mu_set[k], width=width, height=height,
                             angle=angle, fill=False, edgecolor=color[k],
                             linewidth=2, label="Gaussian_{}".format(k))
            ax.add_artist(ell)

    def plot_decision_boundary(self, sample_point, mu_set, cov_set,
                               plot_confidence_interval=False):
        """
        Plot the decision boundary of the Bayes and confidence interval (99.73%)
        ellipses (if plot_confidence_interval=True).

        :param sample_point: [ sample_size * D ], np.array
        :param mu_set: mean set, mean of each Gaussian, [ K * ... ]
        :param cov_set: covariance of each Gaussian, [ K * ... ]
        :param plot_confidence_interval: plot confidence interval or not
        """
        color = ("blue", "orange", "green", "red", "yellow")
        x_max = sample_point[np.argmax(sample_point.T[0])][0]
        x_min = sample_point[np.argmin(sample_point.T[0])][0]
        y_max = sample_point[np.argmax(sample_point.T[1])][1]
        y_min = sample_point[np.argmin(sample_point.T[1])][1]

        plt.rcParams["figure.figsize"] = (10.0, 10.0)
        fig, ax = plt.subplots()

        # plot decision boundary
        x, y = np.meshgrid(np.linspace(x_min - 0.3, x_max + 0.3, 400),
                           np.linspace(y_min - 0.3, y_max + 0.3, 400))

        z = self.predict(np.c_[np.ravel(x), np.ravel(y)])
        z = np.argmax(z, axis=1).reshape(x.shape)
        ax.contourf(x, y, z, self.K - 1, alpha=0.15, colors=color)

        # plot confidence interval
        if plot_confidence_interval:
            self.plot_confidence_interval(mu_set, cov_set, ax, color)

        plt.axis([x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3])
        plt.legend(fontsize=14)
        plt.grid()
        plt.show()
