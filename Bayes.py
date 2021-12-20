import numpy as np
import scipy.stats as st
import matplotlib.patches as mp
import matplotlib.pyplot as plt

class Bayes:
    def __init__(self, mu_set, cov_set):
        self.K = len(mu_set)

        self.mu_set  = mu_set
        self.cov_set = cov_set

    """ Estimator """

    def bayes_inferences(self, sample_point):
        """
        Predict the label of the point in the sample point by bayes inferences.
        Notes that the return value is not the true label like [ 0 0 1 0].
        It's the score of each value [ 10 20 30 5 ]. To get the label, still
        need to take argmax.

        :param sample_point: [ sample_size * D ], np.array
        :return: [ sample_size * K ], np.array
        """
        sample_size = len(sample_point)

        bayes_inferences = np.zeros((sample_size, self.K))
        for n in range(sample_size):
            point = sample_point[n]
            for k in range(self.K):
                mu = self.mu_set[k]
                cov = self.cov_set[k]
                probability = st.multivariate_normal.pdf(point, mu, cov)

                bayes_inferences[n][k] = probability

        return bayes_inferences     # [ N * K ]

    def bayes_accuracy(self, sample_point, sample_label):
        """
        Give a sample point, get the predicting label from the "predict". Then,
        compare the predicting label with the correct label "sample_label", and
        return the accuracy of the prediction

        :param sample_point: [ sample_size * D ], np.array
        :param sample_label: [ sample_size * K ], np.array
        :return: accuracy of the Bayes prediction (float)
        """
        y = np.argmax(self.bayes_inferences(sample_point), axis=1)
        t = np.argmax(sample_label, axis=1)

        return np.sum(y == t) / sample_point.shape[0]

    """ Visualization """

    def plot_confidence_interval(self, ax, color):
        """
        Help function for "plot_decision_boundary", using to plot the confident
        interval ellipse (99.73%) of the normal distribution

        :param ax: axes object of the 'fig'
        :param color: color of the ellipse
        """
        # P Value of Chi-Square->99.73%: 11.8 ; 95.45%: 6.18 ; 68.27%: 2.295
        # P Value from Chi-Square Calculator:
        # https://www.socscistatistics.com/pvalues/chidistribution.aspx
        confidence = 11.8
        for k in range(self.K):
            # calculate eigenvalue and eigenvector
            eigenvalue, eigenvector = np.linalg.eig(self.cov_set[k])
            sqrt_eigenvalue = np.sqrt(np.abs(eigenvalue))

            # calculate all the parameter needed for plotting ellipse
            width = 2 * np.sqrt(confidence) * sqrt_eigenvalue[0]
            height = 2 * np.sqrt(confidence) * sqrt_eigenvalue[1]
            angle = np.rad2deg(np.arccos(eigenvector[0, 0]))

            # plot the ellipse
            ell = mp.Ellipse(xy=self.mu_set[k], width=width, height=height,
                             angle=angle, fill=False, edgecolor=color[k],
                             linewidth=2, label="Gaussian_{}".format(k))
            ax.add_artist(ell)

    def plot_decision_boundary(self, sample_point,
                               plot_confidence_interval=False):
        """
        Plot the decision boundary of the Bayes and confidence interval (99.73%)
        ellipses (if plot_confidence_interval=True).

        :param sample_point: [ sample_size * D ], np.array
        :param plot_confidence_interval: plot confidence interval or not
        """
        color = ("blue", "orange", "green", "red", "yellow")
        x_max = sample_point[np.argmax(sample_point.T[0])][0]
        x_min = sample_point[np.argmin(sample_point.T[0])][0]
        y_max = sample_point[np.argmax(sample_point.T[1])][1]
        y_min = sample_point[np.argmin(sample_point.T[1])][1]

        fig, ax = plt.subplots()

        # plot decision boundary
        x, y = np.meshgrid(np.linspace(x_min - 0.3, x_max + 0.3, 400),
                           np.linspace(y_min - 0.3, y_max + 0.3, 400))

        z = self.bayes_inferences(np.c_[np.ravel(x), np.ravel(y)])
        z = np.argmax(z, axis=1).reshape(x.shape)
        ax.contourf(x, y, z, self.K - 1, alpha=0.15, colors=color)

        # plot confidence interval
        if plot_confidence_interval: self.plot_confidence_interval(ax, color)

        plt.axis([x_min - 0.3, x_max + 0.3, y_min - 0.3, y_max + 0.3])
        plt.legend(fontsize=14)
        plt.grid()
        plt.show()
