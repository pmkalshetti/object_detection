# ----- Import statements ----- #
import numpy as np
import cv2 as cv


def process_data(img, labels,  width_processed, height_processed):
    """Resizes image and labels accordingly.

    Args:
        img (np.ndarray, shape=[height_random, width_random, 3],
            dtype=np.float32): raw RGB image of arbitary width and height.
        labels (np.ndarray, shape=[5,], dtype=np.float32):
            corner points of box.
            [center_x, center_y, width_bbox, height_bbox, idx_category]
            Note: The following is also allowed
            shape=[num_bboxes, 5]
        width_processed (int): width of processed image.
        height_processed (int): height of processed image.

    Returns:
        img_processed (np.ndarray, shape=[height_random, width_random, 3],
            dtype=np.float32): processed RGB image of fixed width and height.
        labels_processed (np.ndarray, shape=[num_bboxes, 5], dtype=np.float32):
            corner points of box as per processed image.
            [center_x, center_y, width_bbox, height_bbox, idx_category]
    """
    # add dim if only 1 label
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=0)

    # resize image
    height_raw, width_raw, _ = img.shape
    img_processed = cv.resize(img, (width_processed, height_processed))

    # resize bunding boxes in labels accordingly
    labels_processed = []
    for label in labels:
        center_x_processed = label[0] * width_processed/width_raw
        center_y_processed = label[1] * height_processed/height_raw
        width_bbox_processed = label[2] * width_processed/width_raw
        height_bbox_processed = label[3] * height_processed/height_raw

        label_processed = np.array([
            center_x_processed,
            center_y_processed,
            width_bbox_processed,
            height_bbox_processed,
            label[4]
        ], dtype=np.float32)
        labels_processed.append(label_processed)
    labels_processed = np.array(labels_processed)

    return img_processed, labels_processed

