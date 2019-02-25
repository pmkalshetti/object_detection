import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_label(labels,
               categories,
               ax):
    """Plot bounding box around object with category names.

    Args:
        labels (np.ndarray, shape=[5,], dtype=np.float32):
            corner points of box.
            [center_x, center_y, width_bbox, height_bbox, idx_category]
            Note: The following is also allowed
            shape=[num_bboxes, 5]
        categories (list of string): object categories.
        ax (matplotlib.axes.Axes): axes object to plot annotation on.
    """
    # add dim if only 1 label
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=0)

    # define plot parameters
    scale_corner = 5
    width_side = 3
    offset_y = (ax.get_ylim()[0] - ax.get_ylim()[1]) * 0.04
    offset_x = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
    colors = plt.cm.jet(np.linspace(0, 1, len(categories)))

    for label in labels:
        color = colors[int(label[4])]

        # corners
        top_left = label[0]-label[2]/2, label[1]-label[3]/2
        top_right = label[0]+label[2]/2, label[1]-label[3]/2
        bottom_left = label[0]-label[2]/2, label[1]+label[3]/2
        bottom_right = label[0]+label[2]/2, label[1]+label[3]/2

        # plot corners
        ax.scatter(top_left[0], top_left[1],
                   color=color, s=scale_corner)
        ax.scatter(top_right[0], top_right[1],
                   color=color, s=scale_corner)
        ax.scatter(bottom_left[0], bottom_left[1],
                   color=color, s=scale_corner)
        ax.scatter(bottom_right[0], bottom_right[1],
                   color=color, s=scale_corner)

        # plot sides
        ax.plot([top_left[0], top_right[0]],
                [top_left[1], top_right[1]],
                color=color, linewidth=width_side)
        ax.plot([top_right[0], bottom_right[0]],
                [top_right[1], bottom_right[1]],
                color=color, linewidth=width_side)
        ax.plot([bottom_right[0], bottom_left[0]],
                [bottom_right[1], bottom_left[1]],
                color=color, linewidth=width_side)
        ax.plot([bottom_left[0], top_left[0]],
                [bottom_left[1], top_left[1]],
                color=color, linewidth=width_side)

        # draw text
        ax.text(
            top_left[0]+offset_x, top_left[1]+offset_y,
            categories[int(label[4])],
            fontsize=15, bbox=dict(facecolor=color, alpha=0.8))
