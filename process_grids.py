import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

import detect_grids_on_array as gdt


def quantify_intensity_per_well(intensity_dict, cutoff=0):
    """

    :param blue_dict: dict of blue squares
    :param cutoff: cutoff to consider signal with cells
    :return: list of intensity values per square and list of square keys above the cutoff
    """

    all = []
    wells_with_cells = []
    for key in intensity_dict.keys():
        value = intensity_dict[key].sum().sum()
        all.append(value)
        if value > cutoff:
            wells_with_cells.append(key)

    return all, wells_with_cells


def expand_square(square, expansion_distance):
    """
    Expand a square to make sure we don't miss any edges
    :param square: array with square coordinates (from find_squares)
    :param expansion_distance: number of pixels to expand the square in all directions
    :return: numpy array of new square with coordinates
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = gdt.get_x_and_y_coords_for_a_square(square)

    min_x = min(x1, x2, x3, x4)
    max_x = max(x1, x2, x3, x4)
    min_y = min(y1, y2, y3, y4)
    max_y = max(y1, y2, y3, y4)

    min_x_new = min_x-expansion_distance
    max_x_new = max_x+expansion_distance

    min_y_new = min_y-expansion_distance
    max_y_new = max_y+expansion_distance

    new_square = np.array([[min_x_new, min_y_new],
                           [min_x_new, max_y_new],
                           [max_x_new, max_y_new],
                           [max_x_new, min_y_new]])
    return new_square


def view_img(img_name):
    """
    Given a individual square file that exists, show the images
    :param img_name:
    :return:
    """
    red = cv.imread(img_name+"red.tiff")
    blue = cv.imread(img_name+"blue.tiff")
    print(os.path.basename(img_name))
    plt.imshow(blue)
    plt.show()
    plt.imshow(red)
    plt.show()
