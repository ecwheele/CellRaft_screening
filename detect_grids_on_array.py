import pandas as pd
import process_grids as pg
import general as gen
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

    elif array_type is None:
        max_length_filter = max_length_filter
        min_length_filter = min_length_filter

    new_dict = dict()
    for key in lengths_dict.keys():
        if (lengths_dict[key] < max_length_filter) & (lengths_dict[key] > min_length_filter):
            new_dict[key] = lengths_dict[key]

    clean_squares_dict = make_new_dict_of_squares(squares, new_dict.keys())
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
    if array_type == "Air_100":
        x_length = 180
        y_length = 180

    else:
        print("Array type not supported")

    averages = df.groupby(by=['x_groups','y_groups']).mean().astype(int)
    df = pd.DataFrame(averages).drop(['index'], axis=1).reset_index()

    new_dict = dict()

    count = 0
    for row in df.iterrows():
        count = count+1
        square = pg.make_square(row[1]['min_x'], row[1]['min_y'],
                      x_length, y_length)
        new_dict[count] = square

    return new_dict


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
    squares, df = make_uniform_squares(y_values.sort_values(by=['x_groups','y_groups']), array_type=array_type)

    return squares, df


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
    x_and_y = gen.get_x_and_y_coords(square)

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
    final_squares = remove_overlapping_squares_v2(new_squares)
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
    sorted_df = gen.make_df_with_square_coords(squares)
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
