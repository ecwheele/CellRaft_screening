import cv2 as cv
import numpy as np
import os

import general as gen
import detect_grids_on_array as gdt


def make_square_coords(image):
    """
    Use to get square coordinates for the whole image (when loading image of a single raft)
    :param image: image loaded with opencv imread
    :return: numpy array defining the square coordinates based on the size of the image
    """

    input_square = np.array([[0,0],
                            [0, image.shape[1]],
                            [image.shape[0], image.shape[1]],
                            [image.shape[0], 0]])
    return input_square


def calculate_ratio(image, input_square, percent_to_crop):
    """
    Calculate the signal ratio between whole image and a smaller version
    :param image: image loaded in with opencv
    :param input_square: square coords for input (from make_square_coords)
    :param percent_to_crop: percent of image to crop in decimal form (0.2 for 20%)
    :return: calculated ratio (in percent), cropped image
    """

    to_expand = (int(image.shape[0]*percent_to_crop))*-1

    whole = image.sum()
    inner = gen.expand_square(input_square, to_expand)

    smaller = gdt.extract_array_from_image(inner, image)
    smaller_sum = smaller.sum()

    ratio = (smaller_sum/whole)*100

    return ratio, smaller


def get_expected_signal(image, percent_to_crop):
    """
    Calculated expected amount of signal based on amount of image cropped
    :param image: image loaded with opencv
    :param percent_to_crop: amount of edge removed in decimal form (0.2 for 20%)
    :return: expected pixel intensity based on area removed
    """
    expected = ((image.shape[0]*(1-percent_to_crop))**2)/(image.shape[0]**2)
    return expected


def calculate_all_ratios(all_imgs, percent_to_crop):
    """
    calculate the ratio of cropped image intensity to full image intensity
    :param all_imgs: list of all images to analyze (full path to storage location)
    :param percent_to_crop: Percent of edges to remove
    :return: dictionary, keys are image name, values are ratio of intensity inside: total intensity
    """

    all_ratios_dict = dict()

    for img_file in all_imgs:

        name = os.path.basename(img_file)
        image = gdt.load_image(img_file)
        input_square = make_square_coords(image)
        to_expand = int(image.shape[0]*percent_to_crop)

        ratio, smaller = calculate_ratio(image, input_square, to_expand)

        all_ratios_dict[name] = ratio

    return all_ratios_dict


def get_expected_signal(image, percent_to_crop):
    """
    Calculate expected intensity based on decrease in area
    :param image: image (loaded with opencv)
    :param percent_to_crop: percent to crop in decimal form
    :return: expected ratio
    """
    expected = ((image.shape[0]*(1-percent_to_crop))**2)/(image.shape[0]**2)
    return expected


def get_cutoffs(all_ratios_dict):
    """
    calculate the cutoff for 1 or 2 standard deviations below the mean
    :param all_ratios_dict: result of calculate_all_ratios
    :return: list [one_std, two_std]
    """
    std = np.std(list(all_ratios_dict.values()))
    mean = np.mean(list(all_ratios_dict.values()))

    one_std = mean - std
    two_std = mean - (2*std)
    return [one_std, two_std]
