import pandas as pd
import find_squares
import cv2 as cv
import numpy.linalg as la
from matplotlib import pyplot as plt
import os


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


def filter_squares_on_side_length(lengths_dict, squares, array_type=None,  
                                 max_length_filter=None, min_length_filter=None):
    """

    :param lengths_dict: Dict of square keys and side lengths
    :param array_type: Air_100 or Air_200
    :param squares: all squares (from find_squares)
    :return: dict of squares only with desired side length
    """

    if array_type == "Air_100":
        max_length_filter = 200
        min_length_filter = 150

    elif array_type == "Air_200":
        max_length_filter = 400
        min_length_filter = 325

    elif array_type == "keyence_10x":
        max_length_filter = 118
        min_length_filter = 80

    elif array_type == None:
        max_length_filter = max_length_filter
        min_length_filter = min_length_filter

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

def make_df_with_minx_miny(squares_dict):
    """
    Makes a df with 2 columns, min_x and min_y
    :param squares_dict:
    :return:
    """
    x_y_dict = get_min_x_and_y_coordinate_from_all_squares(squares_dict)
    df = pd.DataFrame.from_dict(x_y_dict, orient='index')
    df.columns = ['min_x', 'min_y']
    return df



def remove_overlapping_squares(squares_dict, max_distance):
    """
    removes squares that are in close proximity. Max distance from 200um cellraft Air images is 60.
    max distance from 100um cellraft Air images is 20
    :param squares_dict: all squares (from find_squares)
    :param max_distance: max distance to allow separation of datapoints
    :return: list of keys to keep from squares dict
    """

    df = make_df_with_minx_miny(squares_dict)

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


def remove_overlapping_squares_v2(squares_dict):
    """
    removes squares with min_x and min_y that are both within 40 pixels of each other
    :param squares_dict:
    :return:
    """
    df = make_df_with_minx_miny(squares_dict)
    new = df.drop_duplicates(subset=['min_x', 'min_y'])
    x_values = assign_x_in_same_rows(new.sort_values('min_x'))
    y_values = assign_y_in_same_columns(x_values)
    keys_to_keep = list(y_values.drop_duplicates(subset=['x_groups', 'y_groups'])['index'])
    return keys_to_keep


def make_df_with_square_coords(squares_dict):
    """

    :param squares_dict:dictionary of squares
    :return:dataframe with min_x and min_y information
    """
    test_df = pd.DataFrame(index=squares_dict.keys())
    for square in test_df.index:
        to_plot = get_x_and_y_coords_for_plotting(squares_dict[square])
        test_df.ix[square, 'min_x'] = min(to_plot[0])
        test_df.ix[square, 'min_y'] = min(to_plot[1])
    sorted_df = test_df.sort_values(by=['min_x', 'min_y'])
    return sorted_df


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


def assign_well_id(df, filename):
    """

    :param df: result of assign_y_in_same_columns
    :param filename:brightfield filename from CellRaft Air
    :return:dataframe with wellID in a new column
    """
    name = os.path.basename(filename).rstrip("F.tiff")

    for row in df.iterrows():
        row_id = name[0]
        col_id = name[2]
        row_num = int(name[1])
        col_num = int(name[3])

        row_num = row[1]['y_groups'] + row_num
        col_num = row[1]['x_groups'] + col_num
        if row_num > 9:
            row_id = chr(ord(row_id)+1)
            row_num = str(row_num)[1]
        if col_num > 9:
            col_id = chr(ord(col_id)+1)
            col_num = str(col_num)[1]

        new_name = "{}{}{}{}".format(row_id, str(int(row_num)), col_id, str(int(col_num)))
        df.ix[row[0], 'well_id'] = new_name

        if (int(row_num) > 9) | (int(col_num) > 9):
            print("Keep Crying")

    return df


def rename_dict_with_wellid(squares_dict, df_with_well_id):
    """

    :param squares_dict: dictionary of squares
    :param df_with_well_id: dataframe with wellID from assign_well_id
    :return:
    """
    reassignment_dict = dict(zip(df_with_well_id['index'], df_with_well_id['well_id']))
    reassigned = dict((reassignment_dict[key], value) for (key, value) in squares_dict.items())
    return reassigned


def extract_array_from_image(square, image):
    x_and_y = get_x_and_y_coords_for_plotting(square)

    min_x = min(x_and_y[0])
    max_x = max(x_and_y[0])
    min_y = min(x_and_y[1])
    max_y = max(x_and_y[1])
    
    img_array = image[min_y:max_y, min_x:max_x]
    
    return img_array


def master_one_img_gridscan(filename, array_type=None, max_length_filter=None, 
                            min_length_filter=None, distance_filt=None):
    """

    :param filename:name of inputfile from CellRaftAir
    :param array_type:Air_100 or Air_200
    :return: final_squares, img, grey, red_img, blue_img
    """

    if array_type == "Air_100":
        distance_filt = 20
    if array_type == "Air_200":
        distance_filt = 60
    if array_type == "keyence_10x":
        distance_filt = 20

    img, grey = load_and_convert_image_to_gray(filename)
    squares = find_squares.find_squares(img)
    lengths = get_side_lengths_of_all_squares(squares)
    new_squares = filter_squares_on_side_length(lengths, squares, array_type=array_type,  
                                                max_length_filter=max_length_filter, 
                                                min_length_filter=min_length_filter)
#    if array_type == "keyence_10x":
    final_keys = remove_overlapping_squares_v2(new_squares)
#    else:
#       final_keys = remove_overlapping_squares(new_squares, distance_filt)
    red_img, gray = load_and_convert_image_to_gray(filename.replace("F.tiff", "R.tiff"))
    blue_img, gray = load_and_convert_image_to_gray(filename.replace("F.tiff", "B.tiff"))
    final_squares = make_new_dict_of_squares(new_squares, final_keys)
    return final_squares, img, grey, red_img, blue_img


def reassign_squares_on_barcode(squares, filename):
    """

    :param squares: dictionary of squares
    :param filename: filename from CellRaft Air
    :return: dataframe with well assignments
    """
    sorted_df = make_df_with_square_coords(squares)
    df_new = assign_x_in_same_rows(sorted_df)
    df = assign_y_in_same_columns(df_new)
    final = assign_well_id(df, filename)
    return final


def master_all(filename, array_type):
    """

    :param filename: Brightfield F file from CellRaftAir to analyze
    :param array_type:Air_100 or Air_200
    :return: dictionary of squares with the assigned coordinate, brightfield, grey, red, and blue images
    """
    final_squares, img, grey, red_img, blue_img = master_one_img_gridscan(filename, array_type)
    df = reassign_squares_on_barcode(final_squares, filename)
    squares = rename_dict_with_wellid(final_squares, df)
    return squares, img, grey, red_img, blue_img
