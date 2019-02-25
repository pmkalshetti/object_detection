# ----- import statements ----- #
import argparse
import os
import shutil
import glob
import tqdm
import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.plot import plot_label

# ----- script settings ----- #
tf.enable_eager_execution()
np.set_printoptions(precision=2, suppress=True)

# ------ read arguments ----- #
parser = argparse.ArgumentParser()
parser.add_argument("dir_data", help="path to directory containing data")
parser.add_argument("-p", "--dir_plot", metavar="\b",
                    help="path to directory for saving the plots")
parser.add_argument("--process", action="store_true",
                    help="process data")
parser.add_argument("-n", "--no_plot", action="store_true",
                    help="do not display plot")
args = parser.parse_args()

# ----- check dirs ----- #
assert os.path.isdir(args.dir_data), \
    "{} is not a directory".format(args.dir_data)

# check if plots exist already and take necessary action
if args.dir_plot:
    if os.path.exists(args.dir_plot) and os.path.isdir(args.dir_plot):
        while True:
            response = input(
                "{} already exists. Do you want to delete? (y/n): ".
                format(args.dir_plot)).lower().strip()
            if response[0] == 'y':
                print("Existing plot dir will be deleted.")
                shutil.rmtree(args.dir_plot)
                break
            elif response[0] == 'n':
                print("Existing plot dir is kept. Nothing done. Exiting.")
                exit()
            else:
                print("Please enter y or n")
                continue
    if not os.path.isdir(args.dir_plot):
        os.makedirs(args.dir_plot)
        print("Empty directory created at {} for saving plots.".
              format(args.dir_plot))

    # create .gitignore file
    with open("{}/.gitignore".format(args.dir_plot), "w") as file:
        file.write("*\n!.gitignore")

# ----- read data name ----- #

# which data (pascal, hand)
args.dir_data = args.dir_data.rstrip("/")
name_data = os.path.basename(args.dir_data)
names_data = ["VOC2012", "hand"]
assert name_data in names_data, \
    "data name should be one of {}".format(names_data)
print("Reading {} data".format(name_data))


# ----- visualize data ----- #

# define canvas
if args.process:
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
else:
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 1, 1)
fig.canvas.mpl_connect("close_event", lambda event: exit())
if not args.no_plot:
    print("Key press to move ahead. Press q to quit.")

# voc2012
if name_data == names_data[0]:

    categories = ['tvmonitor', 'aeroplane', 'bicycle', 'bird', 'boat',
                  'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train']

    # grid shape
    height_grid, width_grid = 13, 13

    # anchors (in [0,1] space)
    anchors_normalized = np.array(
        [
            [0.09112895, 0.06958421],
            [0.21102316, 0.16803947],
            [0.42625895, 0.26609842],
            [0.25476474, 0.49848],
            [0.52668947, 0.59138947]
        ],
        dtype=np.float32)

    # map from [0,1] to [0, 19] (grid) space
    anchors_grid = anchors_normalized * \
        np.array([height_grid, width_grid], dtype=np.float32)

    # read data
    dir_input = args.dir_data + "/JPEGImages"
    dir_output = args.dir_data + "/Annotations"
    filenames_input = sorted(glob.glob(dir_input+"/*.jpg"))
    filenames_output = sorted(glob.glob(dir_output+"/*.xml"))

    # loop over data
    num_data = len(filenames_input)
    progress_bar = tqdm.tqdm(total=num_data, unit="data", leave=False)
    for idx_data in range(num_data):
        # read input
        img = cv.imread(filenames_input[idx_data])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # read output
        tree = ET.parse(filenames_output[idx_data])
        root = tree.getroot()
        labels = []
        for obj in root.findall("object"):
            # read category
            category = obj.find("name").text
            idx_category = categories.index(category)

            # read bounding box
            bbox = obj.find("bndbox")
            y_min = float(bbox.find("ymin").text)
            x_min = float(bbox.find("xmin").text)
            y_max = float(bbox.find("ymax").text)
            x_max = float(bbox.find("xmax").text)

            # convert from corners to center, shape
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            height_bbox = y_max - y_min
            width_bbox = x_max - x_min

            # concat into label
            label = np.array(
                [x_center, y_center, width_bbox, height_bbox, idx_category],
                dtype=np.float32)
            labels.append(label)
        labels = np.array(labels)

        # process
        if args.process:
            raise NotImplementedError

            ax2.set_title("Processed image")
            ax2.imshow(img_processed)
            plot_label(labels_processed,  categories, ax2)

        # plot
        img_name = os.path.basename(filenames_input[idx_data])[:-4]
        # fig.suptitle(img_name)
        ax1.set_title("Original Image")
        ax1.imshow(img)
        plot_label(labels, categories, ax1)

        fig.canvas.draw()

        # save or show
        if args.dir_plot:
            fig.savefig("{}/{}.svg".format(args.dir_plot, img_name),
                        dpi=300, bbox_inches="tight")
        if not args.no_plot:
            # keypress to move ahead, press q to quit
            flag_close_window = False
            while not flag_close_window:
                flag_close_window = fig.waitforbuttonpress(-1)

        for ax in fig.get_axes():
            ax.clear()
        progress_bar.update(1)
    progress_bar.close()
    if args.dir_plot:
        print("Saved plots in {}".format(args.dir_plot))


# hand
if name_data == names_data[1]:
    raise NotImplementedError
