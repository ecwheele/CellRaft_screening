import detect_squares_v2 as grid
import general as gen
import pandas as pd
from matplotlib import pyplot as plt


def get_square_coords(final_squares, number):
    """

    :param final_squares: result of master_one_img_gridscan
    :param number: number assigned to hit (after filename)
    :return: list of x and y coordinates
    """
    num = 1

    for square in final_squares.keys():
        if num == int(number):
            coords = gen.get_x_and_y_coords(final_squares[square])
            return coords
        num += 1


def check_correct_square(final_squares, hit_filename, red_img, blue_img):
    """

    :param final_squares: result of master_one_img_gridscan
    :param hit_filename: number assigned to hit (after filename)
    :param red_img: result of master_one_img_gridscan
    :param blue_img: result of master_one_img_gridscan
    :return: plots blue and red image and filename to check
    """
    number = hit_filename.split('-')[1]

    num = 1

    for square in final_squares.keys():
        if num == int(number):
            print(hit_filename)
            red = grid.extract_array_from_image(final_squares[square], red_img)
            blue = grid.extract_array_from_image(final_squares[square], blue_img)
            plt.imshow(red)
            plt.show()
            plt.imshow(blue)
            plt.show()
        num += 1


def get_absolute_center(hit_filename, metadata_info, coords):
    """

    :param hit_filename:
    :param metadata_info:
    :param coords:
    :return:
    """

    x_center = (min(coords[0]) + max(coords[0]))/2
    y_center = (min(coords[1]) + max(coords[1]))/2
    absolute_x = metadata_info.loc[(hit_filename.split('-')[0]+".oir").strip("MAX_")]['x_absolute']
    absolute_y = metadata_info.loc[(hit_filename.split('-')[0]+".oir").strip("MAX_")]['y_absolute']

    row = [hit_filename, x_center+absolute_x, y_center+absolute_y]
    return row


def master(data_dir, hits, array_type, check=False):
    """

    :param data_dir:
    :param hits: list of hits
    :param array_type:
    :param check: want to see images called? default is False
    :return: dataframe with hit and coords
    """

    rows = []

    for hit in hits:
        filename = data_dir+"tif/"+hit.split('-')[0]+".oir - C=2.tif"
        number = int(hit.split('-')[1])

        metadata_file = data_dir+'analysis/pos_info.csv'
        metadata = pd.read_csv(metadata_file, index_col=0)

        final_squares, img, red_img, blue_img, df_unused = grid.master_one_img_gridscan(filename,
                                                                                 array_type=array_type,
                                                                                 blur=True)
        coords = get_square_coords(final_squares, number)

        if check is True:
            check_correct_square(final_squares, hit, red_img, blue_img)

        row = get_absolute_center(hit, metadata, coords)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.columns = ['name','x_pos','y_pos']
    df.set_index('name', inplace=True)
    return df
