from Gaussian import Gaussian
from EM import EM
from LNN import LNN
from QNN import QNN

import os
import numpy as np

sample_number = 500
run_time = 1

LNN_activation_func_1 = { 0: LNN.softmax }
LNN_activation_func_2 = { 0: LNN.relu, 1: LNN.softmax }
QNN_activation_func_1 = { 0: QNN.softmax }

optimizer_para = {
    "lr":         0.01,     # float, for all optimizer
    "decay_rate": 0.99,     # float, for optimizer "RMSprop"
    "beta1":      0.9,      # float, for optimizer "Adam"
    "beta2":      0.999,    # float, for optimizer "Adam"
    "iter":       0
}

if not os.path.exists('complex'): os.mkdir('complex')
for D in (3, 2):
    for K in (6, ):
        result = np.zeros([sample_number, 6])
        for S in range(sample_number):
            print(D, K, S)

            LNN_neuron_num_1     = {0: K}
            LNN_neuron_num_2_10  = {0: 10,  1: K}
            LNN_neuron_num_2_100 = {0: 100, 1: K}
            QNN_neuron_num_1     = {0: K}

            while True:
                """ Set mu, cov """
                mu_set = np.array([(np.random.random(D) - 0.5) * 10
                                   for i in range(K)])
                cov_set = []
                for i in range(K):
                    a = np.random.random((D, D)) * 2 - 1
                    cov = np.dot(a, a.T) + np.dot(a, a.T)
                    cov_set.append(cov)

                """ Generate Sample """
                N_k = [np.random.randint(3000, 6000) for k in range(K)]
                train_gaussian = Gaussian(N_k, mu_set, cov_set)
                train_point, train_label = train_gaussian.generate_sample()

                N_k = [np.random.randint(3000, 6000) for k in range(K)]
                test_gaussian = Gaussian(N_k, mu_set, cov_set)
                test_point, test_label = test_gaussian.generate_sample()

                """ Bayes inference """
                bayes_accuracy = test_gaussian.bayes_inference()

                """ EM algorithm """
                em = EM(K)
                em.train(train_point, train_label)
                em_accuracy = em.test(test_point, test_label)

                if 0.005 > bayes_accuracy - em_accuracy >= 0: break

            result[S][0] = bayes_accuracy
            result[S][1] = em_accuracy

            for i in range(run_time):

                qnn = QNN(D, QNN_neuron_num_1, QNN_activation_func_1)
                qnn.train(train_point, train_label, test_point, test_label,
                          optimizer_para)
                accuracy = 0
                for j in range(1, 101):
                    accuracy += qnn.test_accuracy[-j]
                result[S][2] = max(accuracy/100, result[S][2])

                lnn = LNN(D, LNN_neuron_num_1, LNN_activation_func_1)
                lnn.train(train_point, train_label, test_point, test_label,
                          optimizer_para)
                accuracy = 0
                for j in range(1, 101):
                    accuracy += lnn.test_accuracy[-j]
                result[S][3] = max(accuracy/100, result[S][3])

                lnn = LNN(D, LNN_neuron_num_2_10, LNN_activation_func_2)
                lnn.train(train_point, train_label, test_point, test_label,
                          optimizer_para)
                accuracy = 0
                for j in range(1, 101):
                    accuracy += lnn.test_accuracy[-j]
                result[S][4] = max(accuracy/100, result[S][4])

                lnn = LNN(D, LNN_neuron_num_2_100, LNN_activation_func_2)
                lnn.train(train_point, train_label, test_point, test_label,
                          optimizer_para)
                accuracy = 0
                for j in range(1, 101):
                    accuracy += lnn.test_accuracy[-j]
                result[S][5] = max(accuracy/100, result[S][5])

            np.savetxt("complex/D={}, K={}.csv".format(D, K),
                       result, delimiter=",")
