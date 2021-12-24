import numpy as np
import matplotlib.patches as mp


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

    ax.scatter(sample_point[:, 0], sample_point[:, 1], s=5, color=color_set)


def plot_confidence_interval_fill(mu_set, cov_set, ax, color):
    """
    Plot one Gaussian's three confidence interval ellipses.

    :param mu_set: mean of each Gaussian, [ K * D ]
    :param cov_set: covariance of each Gaussian, [ K * D * D ]
    :param ax: axes object of the 'fig'
    :param color: color set. each Gaussian has one corresponding color.
    """
    # P Value of Chi-Square [99.73%: 11.8 ; 95.45%: 6.18 ; 68.27%: 2.295]
    # P Value from Chi-Square Calculator:
    # https://www.socscistatistics.com/pvalues/chidistribution.aspx
    confidence = [11.8, 6.18, 2.295]

    # 99.73% corresponding to the initial transparency
    # 95.45% corresponding to twice of the initial transparency
    # 68.27% corresponding to three times of the initial transparency
    initial_alpha = 0.06

    for k in range(len(mu_set)):
        for i in range(3):
            # calculate eigenvalue and eigenvector
            eigenvalue, eigenvector = np.linalg.eig(cov_set[k])
            sqrt_eigenvalue = np.sqrt(np.abs(eigenvalue))

            # calculate all the parameter needed for plotting ellipse
            width = 2 * np.sqrt(confidence[i]) * sqrt_eigenvalue[0]
            height = 2 * np.sqrt(confidence[i]) * sqrt_eigenvalue[1]
            angle = np.rad2deg(np.arccos(eigenvector[0, 0]))

            # plot the ellipse
            ell = mp.Ellipse(xy=mu_set[k], width=width, height=height,
                             angle=angle, color=color[k])
            ax.add_artist(ell)
            ell.set_alpha(initial_alpha * (i + 1))  # adjust transparency


def plot_confidence_interval_unfill(mu_set, cov_set, ax, color):
    """
    Plot the confident interval ellipse (99.73%) of the normal distribution

    :param mu_set: mean set, mean of each Gaussian, [ K * ... ]
    :param cov_set: covariance of each Gaussian, [ K * ... ]
    :param ax: axes object of the 'fig'
    :param color: color set. each Gaussian has one corresponding color.
    """
    # P Value of Chi-Square [99.73%: 11.8 ; 95.45%: 6.18 ; 68.27%: 2.295]
    # P Value from Chi-Square Calculator:
    # https://www.socscistatistics.com/pvalues/chidistribution.aspx
    confidence = 11.8
    for k in range(len(mu_set)):
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
                         linewidth=2)
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
    x, y = np.meshgrid(np.linspace(x_min - 0.5, x_max + 0.5, 400),
                       np.linspace(y_min - 0.5, y_max + 0.5, 400))

    z = predict(np.c_[np.ravel(x), np.ravel(y)])
    z = np.argmax(z, axis=1).reshape(x.shape)
    ax.contourf(x, y, z, K - 1, alpha=0.15, colors=color)
