import os
import numpy as np


class LNN:
    def __init__(self, dimension, neuron_num, activation_func,
                 load_LNN=False):
        """
        :param dimension: dimension of sample point, int
        :param neuron_num: dictionary, { layer index : number of nodes }
            number of nodes for each layer, including hidden and output layers
        :param activation_func:dictionary, { layer index : function }
            the activation function after each layers' output, including hidden
            and output layers
        :param load_LNN: load parameters of network from file "save/LNN_" or not
        """
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

        # result
        self.train_loss = []
        self.test_loss  = []
        self.train_accuracy = []
        self.test_accuracy  = []

        self.initialize_network(load_LNN)

    def initialize_network(self, load_LNN):
        """
        Initialize five dictionary of the parameters of object "LNN."

        See https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf (English)
            https://arxiv.org/abs/1502.01852 (English)
        about the initialization of parameter weighs and bias.

        :param load_LNN: bool
            If the parameters load from file "save" Notes that the network
            are saved by the help function "save_LNN"
        """
        if len(self.activation_func) != self.L:
            print('Error! Dimension of the "activation_func" not match!')

        for l in range(self.L):
            if l == 0:
                node_from = self.D
            else:
                node_from = self.neuron_num[l - 1]
            node_to   = self.neuron_num[l]

            # sd for initialize weight 'w', parameter of network
            sd = 0.01
            """
            if self.activation_func[l] == self.sigmoid:
                sd = np.sqrt(1 / node_from)
            elif self.activation_func[l] == self.relu:
                sd = np.sqrt(2 / node_from)
            """

            # initialize parameters
            key = 'w' + str(l)
            self.para[key] = sd * np.random.randn(node_from, node_to)
            self.h[key] = np.zeros((node_from, node_to))
            self.m[key] = np.zeros((node_from, node_to))
            self.v[key] = np.zeros((node_from, node_to))

            key = 'b' + str(l)
            self.para[key] = np.zeros((1, node_to))
            self.h[key] = np.zeros((1, node_to))
            self.m[key] = np.zeros((1, node_to))
            self.v[key] = np.zeros((1, node_to))

        if load_LNN: self.load_LNN()

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

         The t is correct label. The first layer of for loop is using to
         traverse entire sample, and the second layer of for loop is using to
         control which y value of that sample will affect the loss value. The
         "y" value will affect the loss value only when it's in the same
         position of correct label.

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
        :return: dictionary, gradient for all the parameters
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
        grad = {}

        # forward
        # a0 -> w0,b0 -> z0 -> a1 -> w1,b1 -> z1 -> a2
        a = {0: sample_point}
        for l in range(self.L):
            z = np.dot(a[l], self.para['w' + str(l)]) + self.para['b' + str(l)]
            a[l + 1] = self.activation_func[l](z)

        # backward
        # da0 <- dw0,db0 <- dz0 <- da1 <- dw1,db1 <- dz1 <- da2
        da = 0
        for l in range(self.L-1, -1, -1):
            if self.activation_func[l] == self.softmax:     # softmax with loss
                dz = (a[l + 1] - sample_label) / len(sample_point)
            elif self.activation_func[l] == self.relu:      # relu
                dz = da * (a[l + 1] != 0)
            else:                                           # sigmoid
                dz = da * (1.0 - a[l + 1]) * a[l + 1]

            grad['w'+str(l)] = np.dot(a[l].T, dz)   # dw
            grad['b'+str(l)] = np.sum(dz, axis=0)   # db
            da = np.dot(dz, self.para['w'+str(l)].T)

        return grad

    """ Four Updating Methods """

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

        return (sample_point + shift) * scale  # [ sample_size * D ], np.array

    def save_result(self, save_result):
        """
        Save result in the file "LNN_result." Notes that the result will be
        saved only when variable of "train", "save_result" is not -1.

        :param save_result: save all result in file "LNN_result" or not, int
            if save_result == -1, mean do not save in file
            if save_result != -1, save in file with name index
                ie: result/LNN_test_accuracy_{save_EM}.csv
        """
        if not os.path.exists('LNN_result'): os.mkdir('LNN_result')
        np.savetxt("LNN_result/train_loss_{}.csv".format(save_result),
                   self.train_loss, delimiter=",")
        np.savetxt("LNN_result/test_loss_{}.csv".format(save_result),
                   self.test_loss, delimiter=",")
        np.savetxt("LNN_result/train_accuracy_{}.csv".format(save_result),
                   self.train_accuracy, delimiter=",")
        np.savetxt("LNN_result/test_accuracy_{}.csv".format(save_result),
                   self.test_accuracy, delimiter=",")

    def save_LNN(self):
        """
        Save all the parameters of the network in the file "save/LNN_". Notes
        that the network will be saved only when variable of "train", "save_LNN"
        is True.
        """
        if not os.path.exists('save'): os.mkdir('save')
        for key in self.para.keys():
            np.savetxt("save/LNN_para_{}.csv".format(key), self.para[key],
                       delimiter=",")
        for key in self.h.keys():
            np.savetxt("save/LNN_h_{}.csv".format(key), self.h[key],
                       delimiter=",")
        for key in self.m.keys():
            np.savetxt("save/LNN_m_{}.csv".format(key), self.m[key],
                       delimiter=",")
        for key in self.v.keys():
            np.savetxt("save/LNN_v_{}.csv".format(key), self.v[key],
                       delimiter=",")

    def load_LNN(self):
        """
        Load all the parameters of the network from the file "save/LNN_".
        Notes that the network's parameters will be initialized by the help
        function only when variable of "initialize_network", "load_LNN" is
        True.
        """
        if not os.path.exists('save'): return
        for l in range(self.L):
            for k in ('w', 'b'):
                key = k+str(l)
                self.para[key] = np.loadtxt(
                    "save/LNN_para_{}.csv".format(key), delimiter=",")
                self.h[key] = np.loadtxt(
                    "save/LNN_h_{}.csv".format(key), delimiter=",")
                self.m[key] = np.loadtxt(
                    "save/LNN_m_{}.csv".format(key), delimiter=",")
                self.v[key] = np.loadtxt(
                    "save/LNN_v_{}.csv".format(key), delimiter=",")

    """ Trainer """

    def result(self, train_point, train_label, test_point, test_label, i):
        """
        Store train result and print state

        :param train_point: [ sample_size * D ], np.array
        :param train_label: [ sample_size * K ], np.array
        :param test_point: [ sample_size * D ], np.array
        :param test_label: [ sample_size * K ], np.array
        :param i: train number/epoch index
        """
        # store result
        train_loss = self.CRE(train_point, train_label)
        train_accuracy = self.accuracy(train_point, train_label)
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)

        test_loss = self.CRE(test_point, test_label)
        test_accuracy = self.accuracy(test_point, test_label)
        self.test_loss.append(test_loss)
        self.test_accuracy.append(test_accuracy)

        # print result
        print('%4d\tL: %10.7f\tA: %7.5f\tL: %10.7f\tA: %7.5f' %
              (i, train_loss, 100 * train_accuracy,
               test_loss, 100 * test_accuracy))

    def train(self, train_point, train_label, test_point, test_label,
              train_number, gradient, optimizer, optimizer_para,
              save_LNN=False, save_result=-1):
        """
        Use a gradient calculator to calculate the gradient of each parameter
        and then use optimizer to update parameters.

        :param train_point: [ sample_size * D ], np.array
        :param train_label: [ sample_size * K ], np.array
        :param test_point: [ sample_size * D ], np.array
        :param test_label: [ sample_size * K ], np.array
        :param train_number: number of iteration
        :param gradient: choose which gradient calculator will be use
        :param optimizer: choose which optimizer will be use
        :param optimizer_para: the parameter dictionary for the optimizer
        :param save_LNN: save parameters in file "save" or not, bool
        :param save_result: save result in file "result" or not, int
        """
        self.result(train_point, train_label, test_point, test_label, 0)
        for i in range(1, train_number+1):
            if i % 2000 == 0:
                self.result(train_point, train_label, test_point, test_label, i)

            # train
            grad = gradient(self, train_point, train_label)
            optimizer(self, grad, optimizer_para)

        # save para/result as .csv
        if save_LNN: self.save_LNN()
        if save_result != -1: self.save_result(save_result)
