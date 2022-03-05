from Gaussian import Gaussian
from Visual import Visual

import numpy as np

sample_number = 500
run_time = 1

for D in (2, 3):
    for K in (6, ):
        """ Set mu, cov """
        mu_set = np.array([(np.random.random(D) - 0.5) * 10 for i in range(K)])
        cov_set = []
        for i in range(K):
            a = np.random.random((D, D)) * 2 - 1
            cov = np.dot(a, a.T) + np.dot(a, a.T)
            cov_set.append(cov)

        """ Generate Sample """
        N_k = [np.random.randint(3000, 6000) for k in range(K)]
        test_gaussian = Gaussian(N_k, mu_set, cov_set)
        test_point, test_label = test_gaussian.generate_sample()

        visual = Visual(test_point, test_label, mu_set, cov_set)
        visual.plot_sample()
