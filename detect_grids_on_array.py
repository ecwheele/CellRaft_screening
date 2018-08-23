import pandas as pd
import numpy as np
import general as gen
from numpy import linalg as la
import find_squares
import cv2 as cv
from matplotlib import pyplot as plt
import os
from typing import Any


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

    elif array_type == "custom_100_20x":
        max_length_filter = 350
        min_length_filter = 280

    elif array_type is None:
        max_length_filter = max_length_filter
        min_length_filter = min_length_filter

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
    if array_type == "Air_100":
        x_length = 180
        y_length = 180

    if array_type == "custom_100_20x":
        x_length = 310
        y_length = 310

    else:
        print("Array type not supported")

    averages = df.groupby(by=['x_groups', 'y_groups']).mean().astype(int)
    df = pd.DataFrame(averages).drop(['index'], axis=1).reset_index()

    new_dict = dict()

    count = -1
    for row in df.iterrows():
        count = count+1
        square = gen.make_square(row[1]['min_x'], row[1]['min_y'],
                                 x_length, y_length)
        new_dict[count] = square

    return new_dict, df


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
    reassignment_dict = dict(zip(df_with_well_id.index, df_with_well_id['well_id']))
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
                            min_length_filter=None):
    """

    :param filename:name of inputfile from CellRaftAir
    :param array_type:Air_100 or Air_200
    :return: final_squares, img, red_img, blue_img
    """

    # if array_type == "Air_100":
    #     distance_filt = 20
    # if array_type == "Air_200":
    #     distance_filt = 60
    # if array_type == "keyence_10x":
    #     distance_filt = 20

    img = load_and_convert_image_to_gray(filename)
    squares = find_squares.find_squares(img)
    lengths = get_side_lengths_of_all_squares(squares)
    new_squares = filter_squares_on_side_length(lengths, squares, array_type=array_type,  
                                                max_length_filter = max_length_filter,
                                                min_length_filter = min_length_filter)

    final_squares, df = remove_overlapping_squares_v2(new_squares, array_type)

    assigned = assign_well_id(df, filename)
    named_squares = rename_dict_with_wellid(final_squares, assigned)

    red_img = load_image(filename.replace("F.tiff", "R.tiff"))
    blue_img = load_image(filename.replace("F.tiff", "B.tiff"))

    return named_squares, img, red_img, blue_img, df


def extract_info_from_named_squares(named_squares, red_img, blue_img,
                                    blue_dict, red_dict, array_type):
    """

    :param named_squares: result of master_one_img_gridscan
    :param red_img: result of master_one_img_gridscan
    :param blue_img: result of master_one_img_gridscan
    :param blue_dict: dictionary to store blue squares
    :param red_dict: dictionary to store red squares
    :param array_type: "Air_100 only supported"
    :return: appends squares that are shrunken into the blue and red dictionaries.
    """

    if array_type == "Air_100":
        expansion_distance = -25

    for square in named_squares.keys():

        shrunk = gen.expand_square(named_squares[square], expansion_distance)
        blue = extract_array_from_image(shrunk, blue_img)
        red = extract_array_from_image(shrunk, red_img)

        blue_dict[square] = blue
        red_dict[square] = red


def find_wells_with_cells(blue_dict, red_dict):

    blues_dict = dict()
    reds_dict = dict()

    for key in blue_dict.keys():
        blue = blue_dict[key].sum().sum()
        red = red_dict[key].sum().sum()

        blues_dict[key] = blue
        reds_dict[key] = red

    blues = list(blues_dict.values())
    reds = list(reds_dict.values())
    blue_cutoff = np.mean(blues) + (0.6*np.std(blues))
    red_cutoff = np.mean(reds) + (0.6*np.std(reds))

    wells_with_cells = []

    for key in blue_dict.keys():
        if (blues_dict[key] > blue_cutoff) & (reds_dict[key] > red_cutoff):
            wells_with_cells.append(key)

    return wells_with_cells


def process_all_files(bright_imgs, array_type, save_dir=None):
    """
    Processes all files that come off the CellRaft Air. Currently only supported for 100um arrays.
    Requires that all files are in the same folder and have the extension R.tiff, B.tiff, F.tiff
    :param bright_imgs: List of all brightfield images. Get with:
                        glob.glob(directory+"*F.tiff")
    :param array_type: "Air_100" is the only one currently supported
    :param save_dir: directory to save result images. Default is to make a new folder inside
                     the directory for the Air images called candidate_wells
    :return: Segments images and keeps wells with cells. Individual red and blue images are
             saved in the location specified by save_dir
    """
    if save_dir is None:

        save_dir = os.path.dirname(bright_imgs[0])+"/candidate_wells/"

    if os.path.isdir(save_dir):
        print("save_dir exists. Create a new folder for this run")

    command = "mkdir {}".format(save_dir)
    os.system(command)

    blue_dict = dict()
    red_dict = dict()

    for image in bright_imgs:
        named_squares, img, red_img, blue_img, df = master_one_img_gridscan(image,
                                                                            array_type = array_type)

        extract_info_from_named_squares(named_squares, red_img, blue_img,
                                        blue_dict, red_dict, array_type)

    wells_with_cells = find_wells_with_cells(blue_dict, red_dict)

    blue_to_save = gen.make_new_dict_of_squares(blue_dict, wells_with_cells)
    red_to_save = gen.make_new_dict_of_squares(red_dict, wells_with_cells)

    for square in blue_to_save.keys():
        cv.imwrite(save_dir+square+"blue.tiff", blue_to_save[square])
        cv.imwrite(save_dir+square+"red.tiff", red_to_save[square])

