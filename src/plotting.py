"""Plotting helpers.

Light blue crosses for the negative class, orange dots for the positive
class, solid black boundary, dashed margins, support vectors circled in navy.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-v0_8-darkgrid')

NEGATIVE_COLOR = '#ADD8E6'
POSITIVE_COLOR = '#FF8C00'
SUPPORT_COLOR = '#00008B'
RESOLUTION = 200


def plot_data(ax, X, y, in_3d=False, heights=None):
    """Scatter the two classes, in the plane or at a given height."""
    negative = y == -1
    positive = y == 1
    if in_3d:
        ax.scatter(X[negative, 0], X[negative, 1], heights[negative],
                   marker='x', c=NEGATIVE_COLOR, s=40, linewidths=1.4,
                   depthshade=False)
        ax.scatter(X[positive, 0], X[positive, 1], heights[positive],
                   marker='o', c=POSITIVE_COLOR, s=32, depthshade=False)
    else:
        ax.scatter(X[negative, 0], X[negative, 1], marker='x',
                   c=NEGATIVE_COLOR, s=55, linewidths=1.6)
        ax.scatter(X[positive, 0], X[positive, 1], marker='o',
                   c=POSITIVE_COLOR, s=45)


def legend_entries(show_margins=True, n_support=0):
    """Legend built by hand, since contours produce no handles."""
    entries = [
        Line2D([], [], marker='x', color=NEGATIVE_COLOR, linestyle='none',
               markersize=8, label='Negative -1'),
        Line2D([], [], marker='o', color=POSITIVE_COLOR, linestyle='none',
               markersize=8, label='Positive +1'),
        Line2D([], [], color='black', linewidth=1.6, label='Decision Boundary'),
    ]
    if show_margins:
        entries.append(Line2D([], [], color='black', linewidth=1.0,
                              linestyle='--', label='Margin'))
    if n_support > 0:
        entries.append(
            Line2D([], [], marker='o', markerfacecolor='none',
                   markeredgecolor=SUPPORT_COLOR, markeredgewidth=1.8,
                   linestyle='none', markersize=10,
                   label='Support Vectors (%d)' % n_support))
    return entries


def plot_margin(model, X, y, title, filename, show_margins=True):
    """Plot the decision boundary, the margins and the support vectors."""
    figure, ax = plt.subplots(figsize=(8, 6))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, RESOLUTION),
                                 np.linspace(y_min, y_max, RESOLUTION))
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # The boundary and margins are the level sets f = 0 and f = +-1.
    values = model.project(grid).reshape(grid_x.shape)

    plot_data(ax, X, y)
    ax.contour(grid_x, grid_y, values, levels=[0], colors='black',
               linewidths=1.6)
    if show_margins:
        ax.contour(grid_x, grid_y, values, levels=[-1, 1], colors='black',
                   linewidths=1.0, linestyles='dashed')

    n_support = 0
    if hasattr(model, 'support'):
        n_support = len(model.support)
        ax.scatter(X[model.support, 0], X[model.support, 1], s=170,
                   facecolors='none', edgecolors=SUPPORT_COLOR, linewidths=1.8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # Equal aspect ratio: the margin is a distance.
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title, fontsize=11)
    ax.legend(handles=legend_entries(show_margins, n_support),
              loc='upper right', fontsize=8, framealpha=1.0)

    figure.savefig('figures/' + filename, dpi=130, bbox_inches='tight')
    plt.close(figure)
    print('  -> figures/' + filename)


def plot_kernel_trick(model_3d, X, y, filename):
    """Lift the data through the explicit map phi(x) = (x1, x2, x1^2 + x2^2).

    model_3d is a linear SVM fitted on phi(X), so no kernel is involved: the
    third coordinate is computed explicitly.
    """
    heights = X[:, 0] ** 2 + X[:, 1] ** 2

    figure = plt.figure(figsize=(14, 5.8))

    ax1 = figure.add_subplot(1, 2, 1)
    plot_data(ax1, X, y)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_aspect('equal')
    ax1.set_title('Input space $\\mathbb{R}^2$: no line can separate these',
                  fontsize=11)

    ax2 = figure.add_subplot(1, 2, 2, projection='3d')
    plot_data(ax2, X, y, in_3d=True, heights=heights)

    plane_x, plane_y = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 12),
        np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 12))
    plane_z = -(model_3d.w[0] * plane_x + model_3d.w[1] * plane_y
                + model_3d.b) / model_3d.w[2]
    ax2.plot_surface(plane_x, plane_y, plane_z, alpha=0.30, color='gray',
                     linewidth=0, shade=False)

    ax2.set_xlabel('$x_1$', labelpad=6)
    ax2.set_ylabel('$x_2$', labelpad=10)
    ax2.set_zticklabels([])
    ax2.view_init(elev=24, azim=-62)
    ax2.set_facecolor('white')
    ax2.set_title('Feature space $\\mathbb{R}^3$: a linear SVM on '
                  '$\\phi(x)$ suffices\n'
                  'vertical axis $= x_1^2 + x_2^2$, no kernel used',
                  fontsize=11)
    ax2.legend(handles=legend_entries(show_margins=False)[:2],
               loc='upper left', fontsize=8)

    figure.subplots_adjust(left=0.03, right=0.97, wspace=0.04)
    figure.savefig('figures/' + filename, dpi=130, bbox_inches='tight')
    plt.close(figure)
    print('  -> figures/' + filename)


def plot_lifted_points(model, X, y, filename):
    """Draw every training point at (x1, x2, f(x)).

    The vertical axis is the output of the model, not a coordinate of its
    feature space, so boundary and margins are horizontal planes.
    """
    heights = model.project(X)

    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    plane_x, plane_y = np.meshgrid(np.linspace(x_min, x_max, 2),
                                   np.linspace(y_min, y_max, 2))

    def draw_plane(ax, level, color, label):
        ax.plot_surface(plane_x, plane_y, np.full(plane_x.shape, level),
                        color=color, alpha=0.18, linewidth=0, shade=False)
        ax.text(x_min, y_min, level, '$f = %s$  ' % label, fontsize=9,
                color=color, ha='right')

    def prepare(ax):
        plot_data(ax, X, y, in_3d=True, heights=heights)
        if hasattr(model, 'support'):
            ax.scatter(X[model.support, 0], X[model.support, 1],
                       heights[model.support], s=110, facecolors='none',
                       edgecolors=SUPPORT_COLOR, linewidths=1.4,
                       depthshade=False)
        ax.set_xlabel('$x_1$', labelpad=4)
        ax.set_ylabel('$x_2$', labelpad=4)
        ax.view_init(elev=16, azim=-72)
        ax.set_facecolor('white')

    figure = plt.figure(figsize=(13.5, 6))

    ax1 = figure.add_subplot(1, 2, 1, projection='3d')
    prepare(ax1)
    draw_plane(ax1, 0.0, 'black', '0')
    ax1.set_title('Each point at its decision value $f(x)$\n'
                  'gaussian kernel, $\\gamma = 2$, $C = 10$', fontsize=11)
    ax1.legend(handles=legend_entries(show_margins=False,
                                      n_support=len(model.support)),
               loc='upper left', fontsize=8)

    ax2 = figure.add_subplot(1, 2, 2, projection='3d')
    prepare(ax2)
    draw_plane(ax2, 1.0, '#B22222', '+1')
    draw_plane(ax2, -1.0, '#1f4e79', '-1')
    ax2.set_title('The margin is the slab between $f = -1$ and $f = +1$\n'
                  'every support vector lies inside it', fontsize=11)

    figure.subplots_adjust(left=0.02, right=0.98, wspace=0.04)
    figure.savefig('figures/' + filename, dpi=130, bbox_inches='tight')
    plt.close(figure)
    print('  -> figures/' + filename)
