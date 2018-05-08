import pandas as pd
import cv2 as cv
import numpy.linalg as la
from matplotlib import pyplot as plt


def load_and_convert_image_to_gray(filename):
    """
    :param filename: Fill path to image file to load
    :return: img file RGB (3D array), and grayscale image (2D array)
    """
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, gray


def get_x_and_y_coords_for_plotting(array):
    """
    Import a value from an array and transform into a list of x and y coords
    :param array: 2D array of square coordinates
    :return: list of x and y coordinates to be used for plotting
    """
    x_vals = [array[0][0], array[1][0], array[2][0], array[3][0]]
    y_vals = [array[0][1], array[1][1], array[2][1], array[3][1]]
    coords = [x_vals, y_vals]
    return coords


def plot_all_squares(array_of_square_coords):
    """
    :param array_of_square_coords: array of square coordinate arrays (from find_squares function)
    :return: Plot of all squares in 2D space
    """
    square_coords = []
    for square in array_of_square_coords:
        coords = get_x_and_y_coords_for_plotting(square)
        plt.scatter(coords[0], coords[1])
        square_coords.append(coords)


def plot_from_dict_of_squares(dict_of_squares):
    """
    this is what my function does
    :param dict_of_squares:
    :return:
    """
    for square in dict_of_squares.keys():
        coords = get_x_and_y_coords_for_plotting(dict_of_squares[square])
        plt.scatter(coords[0], coords[1])


def get_side_lengths_of_all_squares(array_of_square_coords):
    """
    Use this to filter squares with reasonable side lengths
    :param array_of_square_coords: results of find_square
    :return: dictionary of side lengths for each square(only calculating length of one side)
    """
    results_dict = dict()

    for square, num in zip(array_of_square_coords, range(len(array_of_square_coords))):
        x_length = square[0][0]-square[1][0]
        y_length = square[0][1]-square[1][1]
        side_length = la.norm([x_length, y_length])
        results_dict[num] = side_length

    return results_dict


def make_new_dict_of_squares(squares, keys_to_keep):
    """

    :param squares:
    :param keys_to_keep: keys of interest
    :return:
    """
    clean_squares_dict = dict()

    for index in keys_to_keep:
        keep = squares[index]
        clean_squares_dict[index] = keep

    return clean_squares_dict


def filter_squares_on_side_length(lengths_dict, max_length_filter, min_length_filter, squares):
    """

    :param lengths_dict: Dict of square keys and side lengths
    :param max_length_filter: maximum side length to consider
    :param min_length_filter: minimum side length to consider
    :param squares: all squares (from find_squares)
    :return: dict of squares only with desired side length
    """

    new_dict = dict()
    for key in lengths_dict.keys():
        if (lengths_dict[key] < max_length_filter) & (lengths_dict[key] > min_length_filter):
            new_dict[key] = lengths_dict[key]

    clean_squares_dict = make_new_dict_of_squares(squares, new_dict.keys())
    return clean_squares_dict


def get_x_and_y_coords_for_a_square(square_array):
    """
    For a given array, return x and y coordinates
    :param square_array: array with coordinates of one square (from find_squares)
    :return: x1,y1,x2,y2,x3,y3,x4,y4
    """
    x1 = square_array[0][0]
    y1 = square_array[0][1]
    x2 = square_array[1][0]
    y2 = square_array[1][1]
    x3 = square_array[2][0]
    y3 = square_array[2][1]
    x4 = square_array[3][0]
    y4 = square_array[3][1]

    return x1, y1, x2, y2, x3, y3, x4, y4


def get_min_x_and_y_coordinate_from_all_squares(squares_dict):
    """
    Use this to find overlapping squares
    :param squares_dict:
    :return:
    """
    x_y_dict = dict()

    for item in squares_dict.keys():
        x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(squares_dict[item])
        min_x = min(x1, x2, x3, x4)
        min_y = min(y1, y2, y3, y4)
        
        x_y_dict[item] = [min_x, min_y]

    return x_y_dict


def remove_overlapping_squares(squares_dict, max_distance):
    """
    removes squares that are in close proximity. Max distance from cellraft Air images is 60
    :param squares_dict: all squares (from find_squares)
    :param max_distance: max distance to allow separation of datapoints
    :return: list of keys to keep from squares dict
    """

    x_y_dict = get_min_x_and_y_coordinate_from_all_squares(squares_dict)
    df = pd.DataFrame.from_dict(x_y_dict, orient='index')
    df.columns = ['min_x', 'min_y']

    keeps = []
    drops = []

    for index1 in df.index:
        if index1 in drops:
            continue
        else:
            first_x = df.loc[index1, 'min_x']
            first_y = df.loc[index1, 'min_y']
            for index2 in df.index:
                if index2 in drops:
                    continue
                else:
                    second_x = df.loc[index2, 'min_x']
                    second_y = df.loc[index2, 'min_y']
                    x_diff = abs(first_x - second_x)
                    y_diff = abs(first_y - second_y)
                    if (x_diff < max_distance) & (y_diff < max_distance):
                        keep = index1
                        drop = index2

                        keeps.append(keep)
                        drops.append(drop)

    items_to_keep = set(keeps)
    return items_to_keep


def keep_selected_squares_for_extraction(items_to_keep, squares_dict, img):
    """

    :param items_to_keep: set object from remove_overlapping_squares
    :param squares_dict:clean_squares dict after length filtering
    :param img:Image loaded for plotting
    :return: dictionary with squares of interest and plots on top of image
    """

    square_coords_to_extract = dict()

    for item in items_to_keep:
        square_coords_to_extract[item] = squares_dict[item]
        to_plot = get_x_and_y_coords_for_plotting(squares_dict[item])
        plt.scatter(to_plot[0], to_plot[1])
    plt.imshow(img)

    return square_coords_to_extract

