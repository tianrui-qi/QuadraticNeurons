import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

# plt.rcParams["figure.dpi"] = 400

class Visual:
    def __init__(self, sample_point, sample_label, mu_set, cov_set):
        self.K = len(mu_set)
        self.D = len(sample_point[0])

        self.sample_point = sample_point
        self.sample_label = sample_label
        self.mu_set = mu_set
        self.cov_set = cov_set

        self.color = ("orange", "red", "blue", "green", "cyan", "magenta")

        self.legend = [mp.Patch(color=self.color[i],
                                label="Gaussian_{}".format(i))
                       for i in range(self.K)]
        self.legend[0] = mp.Patch(color=self.color[0], label="Background")
        self.x_max = 4
        self.x_min = -5
        self.y_max = 5
        self.y_min = -6
        """
        self.legend = [mp.Patch(color=self.color[i],
                                label="Gaussian_{}".format(i + 1))
                       for i in range(self.K)]
                    
        edge = 0
        self.x_max = sample_point[np.argmax(sample_point.T[0])][0] + edge
        self.x_min = sample_point[np.argmin(sample_point.T[0])][0] - edge
        self.y_max = sample_point[np.argmax(sample_point.T[1])][1] + edge
        self.y_min = sample_point[np.argmin(sample_point.T[1])][1] - edge
        """

    def plot_sample(self):
        ax, fig = None, None

        if self.D == 2:
            fig, ax = plt.subplots()
            """
            plot_confidence_interval_unfill(self.mu_set, self.cov_set,
                                            ax, self.color)
            """
        elif self.D == 3:
            ax = plt.subplot(111, projection='3d')

        if ax is None: return
        plot_scatter(self.sample_point, self.sample_label, ax, self.color)
        plt.legend(handles=self.legend)
        plt.axis([self.x_min, self.x_max, self.y_min, self.y_max])
        plt.grid()

        return fig

    def plot_DB(self, method):
        if self.D != 2: return

        fig, ax = plt.subplots()
        """
        plot_confidence_interval_unfill(self.mu_set, self.cov_set,
                                        ax, self.color)
        """
        plot_decision_boundary(self.K, method.predict, ax, self.color,
                               self.x_min, self.x_max, self.y_min, self.y_max)
        plt.legend(handles=self.legend)
        plt.axis([self.x_min, self.x_max, self.y_min, self.y_max])
        plt.grid()

        return fig


""" Help function for class 'Visual' """


def plot_scatter(sample_point, sample_label, ax, color):
    """
    Plot scatter diagram of the sample. The color of a point is match with its
    label.

    :param sample_point: [ sample_size * D ], np.array
    :param sample_label: [ sample_size * K ], np.array
    :param ax: axes object of the 'fig'
    :param color: color set. each Gaussian has one corresponding color.
    """
    color_set = []
    for n in sample_label:
        color_set.append(color[np.argmax(n)])

    if len(sample_point[0]) == 2:
        ax.scatter(sample_point[:, 0], sample_point[:, 1], s=2, color=color_set)
    elif len(sample_point[0]) == 3:
        ax.scatter(sample_point[:, 0], sample_point[:, 1], sample_point[:, 2],
                   s=2, color=color_set)


def plot_confidence_interval_fill(mu_set, cov_set, ax, color):
    """
    Plot one Gaussian's three confidence interval ellipses.

    :param mu_set: mean of each Gaussian, [ K * D ]
    :param cov_set: covariance of each Gaussian, [ K * D * D ]
    :param ax: axes object of the 'fig'
    :param color: color set. each Gaussian has one corresponding color.
    """
    initial_alpha = [0.18, 0.12, 0.06]

    for k in range(len(mu_set)):
        for i in range(3, 0, -1):
            # calculate eigenvalue and eigenvector
            eigenvalue, eigenvector = np.linalg.eig(cov_set[k])
            sqrt_eigenvalue = np.sqrt(np.abs(eigenvalue))

            # calculate all the parameter needed for plotting ellipse
            width  = 2 * i * sqrt_eigenvalue[0]
            height = 2 * i * sqrt_eigenvalue[1]
            angle  = np.rad2deg(np.arccos(eigenvector[0, 0]))

            # plot the ellipse
            ell = mp.Ellipse(xy=mu_set[k], width=width, height=height,
                             angle=angle, color=color[k])
            ax.add_artist(ell)
            ell.set_alpha(initial_alpha[i-1])  # adjust transparency


def plot_confidence_interval_unfill(mu_set, cov_set, ax, color):
    """
    Plot the confident interval ellipse of the normal distribution

    :param mu_set: mean set, mean of each Gaussian, [ K * ... ]
    :param cov_set: covariance of each Gaussian, [ K * ... ]
    :param ax: axes object of the 'fig'
    :param color: color set. each Gaussian has one corresponding color.
    """
    for k in range(len(mu_set)):
        # calculate eigenvalue and eigenvector
        eigenvalue, eigenvector = np.linalg.eig(cov_set[k])
        sqrt_eigenvalue = np.sqrt(np.abs(eigenvalue))

        # calculate all the parameter needed for plotting ellipse
        width  = 2 * 3 * sqrt_eigenvalue[0]
        height = 2 * 3 * sqrt_eigenvalue[1]
        angle  = np.rad2deg(np.arccos(eigenvector[0, 0]))

        # plot the ellipse
        ell = mp.Ellipse(xy=mu_set[k], width=width, height=height,
                         angle=angle, fill=False, edgecolor=color[k],
                         linewidth=1)
        ax.add_artist(ell)


def plot_decision_boundary(K, predict, ax, color, x_min, x_max, y_min, y_max):
    """
    Plot the decision boundary according to the input variable "predict"

    :param K: number of classification
    :param predict: a function that use to predict the classification
    :param ax: axes object of the "fig"
    :param color: color set. each Gaussian has one corresponding color.
    :param x_min: minimum x value in the "fig"
    :param x_max: maximum x value in the "fig"
    :param y_min: minimum y value in the "fig"
    :param y_max: maximum y value in the "fig"
    :return:
    """
    x, y = np.meshgrid(np.linspace(x_min, x_max, 1000),
                       np.linspace(y_min, y_max, 1000))

    z = predict(np.c_[np.ravel(x), np.ravel(y)])
    z = np.argmax(z, axis=1).reshape(x.shape)
    ax.contourf(x, y, z, K - 1, alpha=0.15, colors=color)
