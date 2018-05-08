import numpy as np
import cv2 as cv


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


def expand_square(square, expansion_distance):
    """
    Expand a square to make sure we don't miss any edges
    :param square: array with square coordinates (from find_squares)
    :param expansion_distance: number of pixels to expand the square in all directions
    :return: numpy array of new square with coordinates
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(square)

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


def grid_in_negative_x_direction(starting_square, x_dist, gap_length):
    """
    make a new square 1 step in the negative x direction
    :param starting_square: array of square coords
    :param x_dist: length of x side of the square (in pixels)
    :param gap_length: gap between squares
    :return: new square 1 step to the negative x direction
    """
    to_move = x_dist+gap_length

    new_square = np.zeros([4, 2])

    for i in range(4):
        old_x = starting_square[i][0]
        y = starting_square[i][1]
        new_x = old_x - to_move
        new_square[i, 0] = new_x
        new_square[i, 1] = y
    starting_square = new_square
    return starting_square


def grid_in_positive_x_direction(starting_square, x_dist, gap_length):
    """
    make a new square 1 step in the positive x direction
    :param starting_square: array of square coords
    :param x_dist: length of x side of the square (in pixels)
    :param gap_length: gap between squares
    :return: new square 1 step to the positive x direction
    """
    to_move = x_dist+gap_length

    new_square = np.zeros([4, 2])

    for i in range(4):
        old_x = starting_square[i][0]
        y = starting_square[i][1]
        new_x = old_x + to_move
        new_square[i, 0] = new_x
        new_square[i, 1] = y
    starting_square = new_square
    return starting_square


def grid_in_positive_y_direction(starting_square, y_dist, gap_length):
    """
    make a new square 1 step in the positive y direction
    :param starting_square: array of square coords
    :param y_dist: length of y side of the square (in pixels)
    :param gap_length: gap between squares
    :return: new square 1 step to the positive y direction
    """
    to_move = y_dist+gap_length

    new_square = np.zeros([4, 2])

    for i in range(4):
        x = starting_square[i][0]
        old_y = starting_square[i][1]
        new_y = old_y + to_move
        new_square[i, 0] = x
        new_square[i, 1] = new_y
    starting_square = new_square
    return starting_square


def grid_in_negative_y_direction(starting_square, y_dist, gap_length):
    """
    make a new square 1 step in the negative y direction
    :param starting_square: array of square coords
    :param y_dist: length of y side of the square (in pixels)
    :param gap_length: gap between squares
    :return: new square 1 step to the negative y direction
    """
    to_move = y_dist+gap_length

    new_square = np.zeros([4, 2])

    for i in range(4):
        x = starting_square[i][0]
        old_y = starting_square[i][1]
        new_y = old_y - to_move
        new_square[i, 0] = x
        new_square[i, 1] = new_y
    starting_square = new_square
    return starting_square


def get_x_and_y_distance_of_square(square):
    """
    getting x and y lengths of the square
    :param square: array with square coords
    :return: x_dist, y_dist
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(square)

    min_x = int(min([x1, x2, x3, x4]))
    min_y = int(min([y1, y2, y3, y4]))
    max_x = int(max([x1, x2, x3, x4]))
    max_y = int(max([y1, y2, y3, y4]))

    x_dist = int(max_x-min_x)
    y_dist = int(max_y-min_y)

    return x_dist, y_dist


def get_gray_matrix_from_square_coords(square, gray_matrix):
    """
    from a given array of square coordinates, extract the image from the grayscale within that array
    :param square: array of square coordinates
    :param gray_matrix: grayscale image matrix
    :return: matrix with grayscale pixel intensities (to be used in cv2)
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(square)
    min_x = int(min([x1, x2, x3, x4]))
    min_y = int(min([y1, y2, y3, y4]))
    max_x = int(max([x1, x2, x3, x4]))
    max_y = int(max([y1, y2, y3, y4]))

    x_vals = list(range(min_x, max_x))
    y_vals = list(range(min_y, max_y))

    new_matrix = np.zeros([len(y_vals), len(x_vals)])

    for x, new_x in zip(x_vals, range(len(x_vals))):
        for y, new_y in zip(y_vals, range(len(y_vals))):
            if (y >= gray_matrix.shape[0]) | (x >= gray_matrix.shape[1]):
                pixel = 0
                new_matrix[new_y, new_x] = pixel
            else:
                pixel = gray_matrix[y, x]
                new_matrix[new_y, new_x] = pixel

    return new_matrix


def make_x_row_of_squares(first_square, gap_distance, gray_img):
    """

    :param first_square: starting point for x line of squares
    :param gap_distance: pixel distance between squares (22 for 10x images on confocal robot)
    :return: dict of x squares across one line
    """
    x_dist, y_dist = get_x_and_y_distance_of_square(first_square)

    x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(first_square)
    first_dict = dict()
    number = 0
    first_dict[number] = first_square

    starting_square = first_square

    while min(x1, x2, x3, x4) > 0:
        number = number+1
        x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(starting_square)
        starting_square = grid_in_negative_x_direction(starting_square,
                                                          x_dist,
                                                          gap_distance)
        first_dict[number] = starting_square

    second_dict = dict()
    x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(first_square)
    starting_square = first_square

    while max(x1, x2, x3, x4) < gray_img.shape[1]:
        number = number+1
        x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(starting_square)
        starting_square = grid_in_positive_x_direction(starting_square, x_dist, gap_distance)
        second_dict[number] = starting_square

    x_line_dict = {**first_dict, **second_dict}

    return x_line_dict


def array_down_from_x_line(x_line, gap_distance):
    """

    :param x_line: output from make_x_row_of_squares
    :param gap_distance:distance between squares
    :return:more squares
    """

    x_dist, y_dist = get_x_and_y_distance_of_square(x_line[0])

    number = len(x_line.keys())
    third_dict = dict()
    for x_grid in x_line.keys():

        starting_square = x_line[x_grid]

        x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(starting_square)

        while min(y1, y2, y3, y4) > 0:
            number = number+1
            starting_square = grid_in_negative_y_direction(starting_square,
                                                              y_dist, gap_distance)
            third_dict[number] = starting_square
            x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(starting_square)


    final_dict = {**third_dict, **x_line}
    return final_dict


def array_up_from_x_line(x_line, gap_distance, gray_img, third_dict):
    x_dist, y_dist = get_x_and_y_distance_of_square(x_line[0])

    number = len(third_dict.keys())
    fourth_dict = dict()
    for x_grid in x_line.keys():

        starting_square = x_line[x_grid]

        x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(starting_square)

        while max(y1,y2,y3,y4) < gray_img.shape[0]:
            number = number+1
            starting_square = grid_in_positive_y_direction(starting_square,
                                                              y_dist, gap_distance)
            fourth_dict[number] = starting_square
            x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(starting_square)

    final_dict = {**third_dict, **fourth_dict}
    return final_dict


def build_rgb_matrix_from_square(square, img):
    """
    get img object from square coordinates
    :param square: square coordinates
    :param img: image
    :return: 3d matrix of image
    """
    img_r = img[:,:,0]
    img_g = img[:,:,1]
    img_b = img[:,:,2]

    new_matrix1 = get_gray_matrix_from_square_coords(square, img_r).astype(int)
    new_matrix2 = get_gray_matrix_from_square_coords(square, img_g).astype(int)
    new_matrix3 = get_gray_matrix_from_square_coords(square, img_b).astype(int)

    result = cv.merge((new_matrix1, new_matrix2, new_matrix3))
    return result


def remove_squares_outside_of_grid(dict_of_squares):
    """
    gets rid of squares with negative x or y values
    :param dict_of_squares: dictionary of squares
    :return: dict of squares without any negatives
    """

    keys_to_drop = []

    for square in dict_of_squares.keys():
        x1, y1, x2, y2, x3, y3, x4, y4 = get_x_and_y_coords_for_a_square(dict_of_squares[square])
        min_x = min(x1, x2, x3, x4)
        min_y = min(y1, y2, y3, y4)
        if (min_x < 0) | (min_y < 0):
            keys_to_drop.append(square)
    final_dict = {square: dict_of_squares[square] for square in dict_of_squares if square not in keys_to_drop}
    return final_dict
