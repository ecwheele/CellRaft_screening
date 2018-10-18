import pandas as pd
import numpy as np
import general as gen
from numpy import linalg as la
import find_squares
import cv2 as cv
import os


# Define square lengths for different arrays. List is [max_length, min_length, uniform_length]
side_lengths_dict = {"Air_100": [200, 150, 180],
                     "Air_200": [400, 325, 375],
                     "keyence_10x": [118, 80, 110],
                     "custom_100_20x_widefield": [350, 280, 310],
                     "custom_100_20x_confocal": [100, 70, 85],
                     "cytosort_100_20x_confocal": [90, 60, 85]}


def load_image(filename):
    """
    :param filename: Fill path to image file to load
    :return: img file
    """
    img = cv.imread(filename, cv.IMREAD_ANYDEPTH)
    return img


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


def filter_squares_on_side_length(lengths_dict, squares, array_type=None,
                                  max_length_filter=None, min_length_filter=None):
    """

    :param lengths_dict: Dict of square keys and side lengths
    :param array_type: Currently support: Air_100, Air_200, keyence_10x, custom_100_20x_widefield,
                        custom_100_20x_confocal, cytosort_100_20x_confocal
    :param squares: all squares (from find_squares)
    :param max_length_filter: max number of pixels for a side length
    :param min_length_filter: min number of pixels for a side length
    :return: dict of squares only with desired side length
    """

    if array_type is None:
        max_length_filter = max_length_filter
        min_length_filter = min_length_filter

    else:
        max_length_filter = side_lengths_dict[array_type][0]
        min_length_filter = side_lengths_dict[array_type][1]

    new_dict = dict()
    for key in lengths_dict.keys():
        if (lengths_dict[key] < max_length_filter) & (lengths_dict[key] > min_length_filter):
            new_dict[key] = lengths_dict[key]

    clean_squares_dict = gen.make_new_dict_of_squares(squares, new_dict.keys())
    return clean_squares_dict


def make_df_with_minx_miny(squares_dict):
    """
    Makes a df with 2 columns, min_x and min_y
    :param squares_dict:
    :return:
    """
    x_y_dict = gen.get_min_x_and_y_coordinate_from_all_squares(squares_dict)
    df = pd.DataFrame.from_dict(x_y_dict, orient='index')
    df.columns = ['min_x', 'min_y']
    df.sort_values(by=['min_x', 'min_y'], inplace=True)
    return df


def make_uniform_squares(df, array_type):
    """

    :param df: output of make_df_with_minx_miny
    :param array_type: Currently support: Air_100, Air_200, keyence_10x, custom_100_20x_widefield,
                        custom_100_20x_confocal, cytosort_100_20x_confocal
    :return:
    """

    length = side_lengths_dict[array_type][2]

    averages = df.groupby(by=['x_groups', 'y_groups']).mean().astype(int)
    df = pd.DataFrame(averages).drop(['index'], axis=1).reset_index()

    new_dict = dict()

    count = -1
    for row in df.iterrows():
        count = count+1
        square = gen.make_square(row[1]['min_x'], row[1]['min_y'],
                                 length, length)
        new_dict[count] = square

    return new_dict, df


def assign_x_in_same_rows(sorted_df):
    """

    :param sorted_df: dataframe sorted on min_x values
    :return: dataframe assigning squares in the same x row
    """
    df = sorted_df.reset_index()

    x_assignments = []
    group = 0
    min_x = df.ix[0, 'min_x']
    for row in sorted_df.iterrows():
        x = row[1]['min_x']
        diff_x = abs(min_x-x)
        if diff_x < 40:
            x_assignments.append(group)

        else:
            group = group + 1
            x_assignments.append(group)

        min_x = x

    df['x_groups'] = x_assignments
    return df.sort_values(by=['x_groups', 'min_y'])


def assign_y_in_same_columns(df):
    """

    :param df: result of assign_x_in_same_rows
    :return:dataframe with squares assigned to columns
    """
    df = df.sort_values(by='min_y').reset_index(drop=True)

    y_assignments = []
    group = 0
    min_y = df.ix[0, 'min_y']
    for row in df.iterrows():
        y = row[1]['min_y']
        diff_y = abs(min_y-y)
        if diff_y < 40:
            y_assignments.append(group)
        else:
            group = group + 1
            y_assignments.append(group)

        min_y = y

    df['y_groups'] = y_assignments
    return df


def remove_overlapping_squares_v2(squares_dict, array_type):
    """
    removes squares with min_x and min_y that are both within 40 pixels of each other
    :param squares_dict: dict with overlapping squares
    :param array_type: "Air_100" is the only one currently supported
    :return: dict of squares and dataframe with x and y col/row assignments
    """
    df = make_df_with_minx_miny(squares_dict)
    new = df.drop_duplicates(subset=['min_x', 'min_y'])
    x_values = assign_x_in_same_rows(new.sort_values('min_x'))
    y_values = assign_y_in_same_columns(x_values)
    squares, df = make_uniform_squares(y_values.sort_values(by=['x_groups', 'y_groups']), array_type=array_type)

    return squares, df


