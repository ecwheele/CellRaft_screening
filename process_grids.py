import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
import os
from scipy import stats

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


def get_y_step(subset_df):
    """
    determine the step for making an artificial grid
    :param subset_df: dataframe with min_x, min_y, x, and y groups for one x row
    :return: distance between squares in the row
    """
    t = list(subset_df['min_y'])
    differences = [abs(j-i) for i,j in zip(t, t[1:])]
    y_step = stats.mode(differences)[0][0]
    return y_step


def fill_missing_xmin_ymin(df_sorted):
    """

    :param df_sorted: dataframe with min_x, min_y, x, and y groups
    :return: new dataframe with missing squares (minx and miny values only)
    """

    subset = df_sorted.loc[df_sorted['x_groups'] == 12]

    y_step = get_y_step(subset)

    new_rows = []

    for group in df_sorted.groupby(by=['x_groups']):
        first_y = min(list(group[1]['y_groups']))
        y_new = first_y
        for row in group[1].iloc[1:].iterrows():
            y = row[1]['y_groups']
            if y == y_new:
                continue

            elif y - y_new == 1:
                y_new = y
                continue

            else:

                num_loops = int(y)-int(y_new)

                for num in range(1, num_loops):
                    to_multiply = num_loops-num
                    new = ['new',row[1]['min_x'], row[1]['min_y']-(y_step*to_multiply),
                          row[1]['x_groups'], y_new+1]
                    new_rows.append(new)
                    y_new = y_new+1
                y_new = y_new+1

    new = df_sorted.append(pd.DataFrame(new_rows, columns = df_sorted.columns))
    new = new.sort_values(by=['x_groups','y_groups'])
    return new


def make_square(min_x, min_y, x_length, y_length):
    """
    format a square properly
    :param min_x: min_x coord
    :param min_y: min_y coord
    :param x_length: x_length of square
    :param y_length: y_length of square
    :return: numpy array of the square
    """
    square = np.array([[min_x, min_y],
             [min_x, min_y + y_length],
             [min_x + x_length, min_y + y_length],
             [min_x + x_length, min_y]])
    return square


def fill_empty_squares_in_array(squares):
    """

    :param squares: dict of squares
    :return: dataframe with missing squares filled in
    """
    df = gdt.make_df_with_square_coords(squares)
    x_rows = gdt.assign_x_in_same_rows(df)
    y_rows = gdt.assign_y_in_same_columns(x_rows)
    df_sorted = y_rows.sort_values(by = ['x_groups','y_groups','min_x','min_y'])
    new = fill_missing_xmin_ymin(df_sorted)
    return new


def add_new_squares_to_dict(squares_dict, df):
    """

    :param squares_dict: dict of squares to add new ones to
    :param df: dataframe from fill_empty_squares_in_array
    :return:
    """

    square_id = max(list(filter(lambda x: x!= 'new', list(df['index']))))

    to_choose = df.loc[(df['index'] != 'new') &
           ((df['x_groups'] == 12) | (df['x_groups'] == 6)) &
           ((df['y_groups'] == 13) | (df['y_groups'] == 4))]
    to_choose = list(to_choose['index'])[0]

    squares_to_make = df.loc[df['index'] == 'new']

    to_plot = gdt.get_x_and_y_coords_for_plotting(squares_dict[to_choose])
    x_length = max(to_plot[0]) - min(to_plot[0])
    y_length = max(to_plot[1]) - min(to_plot[1])


    for row in squares_to_make.iterrows():
            square_id = square_id+1
            min_x = row[1]['min_x']
            min_y = row[1]['min_y']

            array = make_square(min_x, min_y, x_length, y_length)
            squares_dict[square_id] = array

    return squares_dict
