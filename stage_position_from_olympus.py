import re
import pandas as pd


def get_setup_parameters(matl_file):
    """

    :param matl_file: full path to output from olympus run (matl.omp2info)
    :return: dictionary with parameters from run
    """

    results = dict()

    file = open(matl_file, "r")

    for line in file.readlines():

        if 'matl:overlap' in line:
            results['overlap'] = int(re.search('<matl:overlap>(.*)</matl:overlap>', line).group(1))

        if 'marker:coordinates' in line:
            results['x_origin'] = int(line.split('"')[5])
            results['y_origin'] = int(line.split('"')[7])

        if '<matl:areaWidth>' in line:
            results['x_step'] = int(re.search('<matl:areaWidth>(.*)</matl:areaWidth>', line).group(1))

        if '<matl:areaHeight>' in line:
            results['y_step'] = int(re.search('<matl:areaHeight>(.*)</matl:areaHeight>', line).group(1))

    return results


def get_image_name_and_steps(matl_file):
    """

    :param matl_file: full path to output from olympus run (matl.omp2info)
    :return: dataframe with x and y step listed for each file
    """

    rows = []

    with open(matl_file) as file:
        for line in file:
            if 'matl:image' in line:
                filename = re.search('<matl:image>(.*)</matl:image>', line).group(1)
                nextline = next(file)
                x_step = re.search('<matl:xIndex>(.*)</matl:xIndex>', nextline).group(1)
                nextline = next(file)
                y_step = re.search('<matl:yIndex>(.*)</matl:yIndex>', nextline).group(1)

                row = [filename, x_step, y_step]

                rows.append(row)

    all_steps = pd.DataFrame(rows)
    all_steps.columns = ['filename', 'x_step', 'y_step']
    all_steps.set_index('filename', inplace=True)

    return all_steps


def get_absoltue_coords_per_image(params_dict, steps_df):
    """

    :param params_dict: output of get_setup_parameters
    :param steps_df: output of get_image_name_and_steps
    :return: df with absolute x and y position per image
    """

    new_coords = []

    for row in steps_df.iterrows():
        name = row[0]
        x_new = (params_dict['x_origin'] +
                 (int(row[1]['x_step'])*params_dict['x_step']) -
                 (int(row[1]['x_step'])*params_dict['overlap']))

        y_new = (params_dict['y_origin'] +
             (int(row[1]['y_step'])*params_dict['y_step']) -
             (int(row[1]['y_step'])*params_dict['overlap']))

        new_coords.append([name, x_new, y_new])

    new = pd.DataFrame(new_coords)
    new.columns = ['filename','absolute_x','absolute_y']
    new.set_index('filename', inplace=True)

    file_coords = steps_df.join(new)

    return file_coords


def master(matl_file, filename_to_save=None):
    """

    :param matl_file: full path to output from olympus run (matl.omp2info)
    :param filename_to_save: location to save resulting dataframe as csv
    :return:
    """
    params_dict = get_setup_parameters(matl_file)
    all_steps = get_image_name_and_steps(matl_file)
    final_df = get_absoltue_coords_per_image(params_dict, all_steps)

    if filename_to_save is not None:
        final_df.to_csv(filename_to_save)

    return final_df
