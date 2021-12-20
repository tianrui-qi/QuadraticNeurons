import random
import numpy as np
import matplotlib.patches as mp
import matplotlib.pyplot as plt


class Gaussian:
    def __init__(self, mean, cov, sample_point=None, sample_label=None):
        self.mean = mean    # mean of a Gaussian
        self.cov  = cov     # covariance of a Gaussian

        self.sample_point = sample_point
        self.sample_label = sample_label

    def generate_point(self, n):
        """
        Generate n number of sample point according to the Gaussian Distribution

        :param n: number of sample point that need to generate
        :return: the list of sample point that generate
        """
        self.sample_point = np.random.multivariate_normal(self.mean, self.cov,
                                                          n)
        return self.sample_point

    def set_label(self, label, k):
        """
        Generate a 1*k matrix that standard for the label of the Gaussian

        Example:
        label = 0, k = 4
            self.label = [1, 0, 0, 0]
        label = 3, k = 4
            self.label = [0, 0, 0, 1]

        :param label: label index of this Gaussian
        :param k: total number of Gaussian
        :return: a 1*k matrix that standard for the label of the Gaussian
        """
        self.sample_label = np.zeros(k)
        self.sample_label[label] = 1
        return self.sample_label

    def plot_gaussian(self, ax, color, legend, plot_confidence_interval=False):
        """
        Plot scatter diagram of the Gaussian
        Plot three confidence interval ellipses of the Gaussian (if
        "plot_confidence_interval" = True)
        Notes that transparency decrease (alhpa increase) as confident interval
        decrease

        :param ax: axes object of the 'fig'
        :param color: color of the ellipse
        :param legend: label of the graph
        :param plot_confidence_interval: plot confidence interval or not
        """
        ax.scatter(self.sample_point[:, 0], self.sample_point[:, 1],
                   s=5, color=color, label=legend)

        if not plot_confidence_interval: return

        # P Value of Chi-Square [99.73%: 11.8 ; 95.45%: 6.18 ; 68.27%: 2.295]
        # P Value from Chi-Square Calculator:
        # https://www.socscistatistics.com/pvalues/chidistribution.aspx
        confidence = [11.8, 6.18, 2.295]

        # 99.73% corresponding to the initial transparency
        # 95.45% corresponding to twice of the initial transparency
        # 68.27% corresponding to three times of the initial transparency
        initial_alpha = 0.06

        for i in range(3):
            # calculate eigenvalue and eigenvector
            eigenvalue, eigenvector = np.linalg.eig(self.cov)
            sqrt_eigenvalue = np.sqrt(np.abs(eigenvalue))

            # calculate all the parameter needed for plotting ellipse
            width  = 2 * np.sqrt(confidence[i]) * sqrt_eigenvalue[0]
            height = 2 * np.sqrt(confidence[i]) * sqrt_eigenvalue[1]
            angle = np.rad2deg(np.arccos(eigenvector[0, 0]))

            # plot the ellipse
            ell = mp.Ellipse(xy=self.mean, width=width, height=height,
                             angle=angle, color=color)
            ax.add_artist(ell)
            ell.set_alpha(initial_alpha * (i + 1))  # adjust transparency


class Gaussian_Set:
    def __init__(self, mu_set, cov_set):
        self.K = len(mu_set)

        self.gaussian_set = self._initialize_gaussian_set(mu_set, cov_set)
        self.mu_set = mu_set
        self.cov_set = cov_set

        self.sample_point = None   # [ N * D ]
        self.sample_label = None   # [ N * K ]

    def _initialize_gaussian_set(self, mu_set, cov_set):
        """
        Initialize K number of Gaussian by object "Gaussian."
        Set their label at the same time, in order.

        :param mu_set: mean set, mean of each Gaussian, [ K * ... ]
        :param cov_set: covariance of each Gaussian, [ K * ... ]
        :return: set of "Gaussian" object, [ K ]
        """
        gaussian_set = []  # [ K ]
        for i in range(self.K):
            # initialize the Gaussian and set label
            gaussian = Gaussian(mu_set[i], cov_set[i])
            gaussian.set_label(i, self.K)
            gaussian_set.append(gaussian)

        return gaussian_set  # [ K ]

    def generate_point_set(self, N_k):
        """
        Collect all the data points and label from all Gaussian's sample and
        shuffle (random) them at the same time.
        Split sample data into two part: test and train.

        :param N_k: number of sample for each Gaussian
        :return: sample_point: [ N * D ], np.array
                 sample_label: [ N * K ], np.array
        """
        sample_set = []
        for k_index in range(self.K):
            point = self.gaussian_set[k_index].generate_point(N_k)
            label = self.gaussian_set[k_index].sample_label
            for n_k in range(N_k):
                sample_set.append((point[n_k], label))
        random.shuffle(sample_set)

        self.sample_point = np.array( [x[0] for x in sample_set] )
        self.sample_label = np.array( [x[1] for x in sample_set] )

        return self.sample_point, self.sample_label

    """ Visualization """

    def plot_gaussian_set(self, plot_confidence_interval=False):
        """
        Plot all Gaussian's scatter diagram and three confidence interval
        ellipses (if plot_confidence_interval=True).

        :param plot_confidence_interval: plot confidence interval or not
        """
        color = ("blue", "orange", "green", "red", "yellow")
        x_max = self.sample_point[np.argmax(self.sample_point.T[0])][0]
        x_min = self.sample_point[np.argmin(self.sample_point.T[0])][0]
        y_max = self.sample_point[np.argmax(self.sample_point.T[1])][1]
        y_min = self.sample_point[np.argmin(self.sample_point.T[1])][1]

        fig, ax = plt.subplots()

        for i in range(self.K):
            legend = "Gaussian_{}".format(i)
            self.gaussian_set[i].plot_gaussian(ax, color[i], legend,
                                               plot_confidence_interval)

        plt.axis([x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3])
        plt.legend(fontsize=14)
        plt.grid()
        plt.show()
