from Gaussian import Gaussian
from LNN import LNN
from QNN import QNN
from Visual import Visual

import os
import numpy as np

sample_number = 1000
EM_train_number = 1000
run_number = 20
NN_train_number = 10000

LNN_activation_func_1 = { 0: LNN.softmax }
LNN_activation_func_2 = { 0: LNN.relu, 1: LNN.softmax }
QNN_activation_func_1 = { 0: QNN.softmax }
QNN_activation_func_2 = { 0: QNN.relu, 1: QNN.softmax }

optimizer_para = {
    "lr":         0.01,     # float, for all optimizer
    "decay_rate": 0.99,     # float, for optimizer "RMSprop"
    "beta1":      0.9,      # float, for optimizer "Adam"
    "beta2":      0.999,    # float, for optimizer "Adam"
    "iter":       0
}

for D in (2, 3):
    for K in (5, 6, 7, 8):
        for S in range(sample_number):
            file = "D={}, K={}".format(D, K)
            if not os.path.exists('complex'): os.mkdir('complex')

            LNN_neuron_num_1     = {0: K}
            LNN_neuron_num_2_10  = {0: 10,  1: K}
            LNN_neuron_num_2_100 = {0: 100, 1: K}
            LNN_neuron_num_3_100 = {0: 100, 1: 50, 2: K}
            LNN_neuron_num_4_100 = {0: 100, 1: 50, 2: 50, 3: K}
            QNN_neuron_num_1     = {0: K}
            QNN_neuron_num_2_10  = {0: 10, 1: K}

            N_k = [np.random.randint(1000, 3000) for k in range(K)]

            # Generate Sample
            mu_set = np.array([(np.random.random(D) - 0.5) * 15 for i in range(K)])
            cov_set = []
            for i in range(K):
                a = np.random.random((D, D)) * 2 - 1
                cov = np.dot(a, a.T) + np.dot(a, a.T)
                cov_set.append(cov)
