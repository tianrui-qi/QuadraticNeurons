import time
import numpy as np


class NN:
    def __init__(self, dimension, neuron_num, activation_func, NN_type="QNN"):
        """
        :param dimension: dimension of sample point, int
        :param neuron_num: dictionary, { layer index : number of nodes }
            number of nodes for each layer, including hidden and output layers
        :param activation_func:dictionary, { layer index : function }
            the activation function after each layers' output, including hidden
            and output layers
        :param NN_type: which type of neural network: "QNN" or "LNN"
        """
        self.NN_type = NN_type

        # basic dimension parameter
        self.L = len(neuron_num)  # number of hidden & output layer
        self.D = dimension  # dimension of sample data point
        self.K = neuron_num[self.L - 1]  # number of Gaussian / classifications

        # network parameter
        self.neuron_num = neuron_num
        self.activation_func = activation_func
        self.para = {}  # weight and bias

        # optimizer parameter
        self.h = {}  # for optimizer "AdaGrad", "RMSprop"
        self.m = {}  # for optimizer "Adam"
        self.v = {}  # for optimizer "Adam"
        self.opt_para = {
            "lr": 0.01,  # float, for all optimizer
            "decay_rate": 0.99,  # float, for optimizer "RMSprop"
            "beta1": 0.9,  # float, for optimizer "Adam"
            "beta2": 0.999,  # float, for optimizer "Adam"
            "iter": 0  # int, for optimizer "Adam"
        }

        self.initialize()  # initialize para, h, m, v

        # result
        self.iteration = []
        self.train_time = 0
        self.valid_loss = []

    def _initialize_LNN(self):
        """
        Initialize parameters for LNN

        See https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf (English)
            https://arxiv.org/abs/1502.01852 (English)
        about the initialization of parameter weighs and bias.
        """
        if len(self.activation_func) != self.L:
            print('Error! Dimension of the "activation_func" not match!')

        for l in range(self.L):
            if l == 0:
                node_from = self.D
            else:
                node_from = self.neuron_num[l - 1]
            node_to = self.neuron_num[l]

            # sd for initialize weight 'w', parameter of network
            sd = 0.01
            """ # He initialization
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

    def _initialize_QNN(self):
        """
        Initialize parameters for QNN

        See https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf (English)
            https://arxiv.org/abs/1502.01852 (English)
        about the initialization of parameter weighs and bias.
        """
        if len(self.activation_func) != self.L:
            print('Error! Dimension of the "activation_func" not match!')

        for l in range(self.L):
            if l == 0:
                node_from = self.D
            else:
                node_from = self.neuron_num[l - 1]
            node_to = self.neuron_num[l]

            # sd for initialize weight 'w', parameter of network
            sd = 0.01
            """ # He initialization
            if self.activation_func[l] == self.sigmoid:
                sd = np.sqrt(1 / node_from)
            elif self.activation_func[l] == self.relu:
                sd = np.sqrt(2 / node_from)
            """

            # initialize parameters
            for j in ('r', 'g', 'b'):
                key = 'w' + j + str(l)
                self.para[key] = sd * np.random.randn(node_from, node_to)
                self.h[key] = np.zeros((node_from, node_to))
                self.m[key] = np.zeros((node_from, node_to))
                self.v[key] = np.zeros((node_from, node_to))

                key = 'b' + j + str(l)
                self.para[key] = np.zeros((1, node_to))
                self.h[key] = np.zeros((1, node_to))
                self.m[key] = np.zeros((1, node_to))
                self.v[key] = np.zeros((1, node_to))

    def initialize(self):
        if self.NN_type == "LNN": return self._initialize_LNN()
        if self.NN_type == "QNN": return self._initialize_QNN()

    def load(self, para, h, m, v):
        self.para, self.h, self.m, self.v = para, h, m, v

    """ Activation Functions """

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

    """ Trainer """

    def _gradient_LNN(self, point, label):
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

        :param point: [ sample_size * D ], np.array
        :param label: [ sample_size * K ], np.array
        :return: dictionary, gradient for all the parameter
        """
        grad = {}

        # forward
        # a0 -> w0,b0 -> z0 -> a1 -> w1,b1 -> z1 -> a2
        a = {0: point}
        for l in range(self.L):
            z = np.dot(a[l], self.para['w' + str(l)]) + self.para['b' + str(l)]
            a[l + 1] = self.activation_func[l](z)

        # backward
        # da0 <- dw0,db0 <- dz0 <- da1 <- dw1,db1 <- dz1 <- da2
        da = 0
        for l in range(self.L - 1, -1, -1):
            if self.activation_func[l] == self.softmax:  # softmax with loss
                dz = (a[l + 1] - label) / len(point)
            elif self.activation_func[l] == self.relu:  # relu
                dz = da * (a[l + 1] != 0)
            else:  # sigmoid
                dz = da * (1.0 - a[l + 1]) * a[l + 1]

            grad['w' + str(l)] = np.dot(a[l].T, dz)  # dw
            grad['b' + str(l)] = np.sum(dz, axis=0)  # db
            da = np.dot(dz, self.para['w' + str(l)].T)

        return grad

    def _gradient_QNN(self, point, label):
        grad = {}

        # forward
        a = {0: point}
        zr = {}
        zg = {}
        for l in range(self.L):
            zr[l] = np.dot(a[l], self.para['wr' + str(l)]) + \
                    self.para['br' + str(l)]
            zg[l] = np.dot(a[l], self.para['wg' + str(l)]) + \
                    self.para['bg' + str(l)]
            zb = np.dot(a[l] ** 2, self.para['wb' + str(l)]) + \
                 self.para['bb' + str(l)]
            z = np.multiply(zr[l], zg[l]) + zb
            a[l + 1] = self.activation_func[l](z)

        # backward
        da = 0
        for l in range(self.L - 1, -1, -1):
            if self.activation_func[l] == self.softmax:  # softmax with loss
                dz = (a[l + 1] - label) / len(point)
            elif self.activation_func[l] == self.relu:  # relu
                dz = da * (a[l + 1] != 0)
            else:  # sigmoid
                dz = da * (1.0 - a[l + 1]) * a[l + 1]

            dzr = dz * zg[l]
            dzg = dz * zr[l]
            dzb = dz

            grad['br' + str(l)] = np.sum(dzr, axis=0)
            grad['bg' + str(l)] = np.sum(dzg, axis=0)
            grad['bb' + str(l)] = np.sum(dzb, axis=0)

            grad['wr' + str(l)] = np.dot(a[l].T, dzr)
            grad['wg' + str(l)] = np.dot(a[l].T, dzg)
            grad['wb' + str(l)] = np.dot(np.square(a[l].T), dzb)

            dar = np.dot(dzr, self.para['wr' + str(l)].T)
            dag = np.dot(dzg, self.para['wg' + str(l)].T)
            dab = np.dot(dzb, self.para['wb' + str(l)].T) * a[l]

            da = dar + dag + dab + dab

        return grad

    def gradient(self, point, label):
        if self.NN_type == "LNN": return self._gradient_LNN(point, label)
        if self.NN_type == "QNN": return self._gradient_QNN(point, label)

    def _SGD(self, grad):
        """
        "Stochastic Gradient Descent"

        Update all parameters including weighs and biases

        :param grad: dictionary
        """
        for key in grad.keys():
            self.para[key] -= self.opt_para["lr"] * grad[key]

    def _AdaGrad(self, grad):
        """
        "Adaptive Gradient Algorithm", an improvement basis on "SGD" above

        Update all parameters including weighs and biases
        Can adjust learning rate by h
        At beginning, since the sum of squares of historical gradients is
        smaller, h += grads * grads the learning rate is high at first.
        As the sum of squares of historical gradients become larger, the
        learning rate will decrease

        :param grad:dictionary
        """
        delta = 1e-7  # avoid divide zero
        for key in grad.keys():
            self.h[key] += np.square(grad[key])
            self.para[key] -= self.opt_para["lr"] * grad[key] / \
                              (np.sqrt(self.h[key]) + delta)

    def _RMSprop(self, grad):
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
        """
        delta = 1e-7  # avoid divide zero
        for key in grad.keys():
            self.h[key] *= self.opt_para["decay_rate"]
            self.h[key] += (1.0 - self.opt_para["decay_rate"]) * \
                           np.square(grad[key])
            self.para[key] -= self.opt_para["lr"] * grad[key] / \
                              (np.sqrt(self.h[key]) + delta)

    def _Adam(self, grad):
        """
        "Adaptive Moment Estimation", an improvement basis on "RMSprop" above
        and momentum

        See "https://arxiv.org/abs/1412.6980v8" (English)

        :param grad: dictionary
        """
        self.opt_para["iter"] += 1
        lr_t = self.opt_para["lr"] * \
               np.sqrt(1.0 - self.opt_para["beta2"] ** self.opt_para["iter"]) / \
               (1.0 - self.opt_para["beta1"] ** self.opt_para["iter"])
        delta = 1e-7  # avoid divide zero
        for key in grad.keys():
            self.m[key] += (1.0 - self.opt_para["beta1"]) * \
                           (grad[key] - self.m[key])
            self.v[key] += (1.0 - self.opt_para["beta2"]) * \
                           (grad[key] ** 2 - self.v[key])
            self.para[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + delta)

    def optimizer(self, optimizer, grad):
        if optimizer == "Adam":
            self._Adam(grad)
        elif optimizer == "RMSprop":
            self._RMSprop(grad)
        elif optimizer == "AdaGrad":
            self._AdaGrad(grad)
        else:
            self._SGD(grad)

    def train(self, train_point, train_label,
              valid_point=None, valid_label=None,
              opt_para=None, optimizer="Adam", epoch=20000, stop_point=500):
        """
        Use a gradient calculator to calculate the gradient of each parameter
        and then use optimizer to update parameters.

        Args:
            train_point: [ sample_size * D ], np.array
            train_label: [ sample_size * K ], np.array
            opt_para: the dictionary store parameter for the optimizer
            valid_point: [ sample_size * D ], np.array
            valid_label: [ sample_size * K ], np.array
            optimizer: choose optimizer: "Adam", "RMSprop", "AdaGrad", "SGD"
            epoch: number of iteration
            stop_point: stop training after "stop_point" number of
            iteration such that the accuracy of validation set does not increase
        """
        if opt_para is not None: self.opt_para = opt_para

        stop_track = 0
        loss_max = 1000
        for i in range(epoch):
            if stop_point <= stop_track: break

            # Main part ========================================================
            begin = time.time()

            self.optimizer(optimizer, self.gradient(train_point, train_label))

            end = time.time()
            self.train_time += end - begin
            # ==================================================================

            """ Recording """

            step_size = 100
            if i % step_size != 0: continue

            self.iteration.append(i)
            if valid_label is not None:
                self.valid_loss.append(self.CRE(valid_point, valid_label))

            """ Early Stopping """

            if valid_label is None: continue
            if self.valid_loss[-1] < loss_max:
                stop_track = 0
                loss_max = self.valid_loss[-1]
            else:
                stop_track += step_size

    """ Estimator """

    def _predict_LNN(self, point):
        a = point  # [ N * K ], np.array
        for l in range(self.L):
            z = np.dot(a, self.para['w' + str(l)]) + self.para['b' + str(l)]
            a = self.activation_func[l](z)
        return a  # [ N * K ], np.array

    def _predict_QNN(self, point):
        a = point  # [ N * K ], np.array
        for l in range(self.L):
            zr = np.dot(a, self.para['wr' + str(l)]) + self.para['br' + str(l)]
            zg = np.dot(a, self.para['wg' + str(l)]) + self.para['bg' + str(l)]
            zb = np.dot(a ** 2, self.para['wb' + str(l)]) + self.para['bb' + str(l)]
            z = np.multiply(zr, zg) + zb
            a = self.activation_func[l](z)
        return a  # [ N * K ], np.array

    def predict(self, point):
        if self.NN_type == "LNN": return self._predict_LNN(point)
        if self.NN_type == "QNN": return self._predict_QNN(point)

    def CRE(self, point, label):
        """
        Return the cross entropy error (CRE).
        """
        # 1. check the input data
        if len(point[0]) != self.D or len(label[0]) != self.K: return 0

        # 2. compute the loss
        y = self.predict(point)  # predict label
        t = label  # actual label
        return -(np.sum(np.multiply(t, np.log(y + 1e-10))) / point.shape[0])

    def test(self, point, label):
        t = np.argmax(label, axis=1)                # actual label
        y = np.argmax(self.predict(point), axis=1)  # predict label

        accuracy = np.sum(y == t) / len(label)
        precision = []
        recall = []
        for k in range(self.K):     # find the precision of each cluster
            TP = 0  # Predict class == k, Actual class == k
            FP = 0  # Predict class == k, Actual class != k
            TN = 0  # Predict class != k, Actual class != k
            FN = 0  # Predict class != k, Actual class == k
            for n in range(len(point)):
                if y[n] == k and t[n] == k: TP += 1
                if y[n] == k and t[n] != k: FP += 1
                if y[n] != k and t[n] != k: TN += 1
                if y[n] != k and t[n] == k: FN += 1

            if TP + FP == 0: precision.append(0)  # avoid divide zero
            else: precision.append(TP / (TP + FP))

            if TP + FN == 0: recall.append(0)  # avoid divide zero
            else: recall.append(TP / (TP + FN))

        return [accuracy, precision, recall]
