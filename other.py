import special
from Visual import Visual
from special import *

import numpy as np


mu_set  = np.array([[-3.0, 1.0],              [-1.0, -3.0]])
cov_set = np.array([[[1.0, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 1.0]]])



train_point, train_label, test_point, test_label, sample_point, sample_label = \
    special.load_sample(12)

visual = Visual(sample_point, sample_label, mu_set, cov_set, bg=True)
visual.plot_sample().savefig("special/fig/{}_sample".format(1111))
