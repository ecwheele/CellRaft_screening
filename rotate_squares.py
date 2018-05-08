import numpy as np
import numpy.linalg as la
import math

import detect_grids_on_array as gdt


def get_vectors_from_square(one_square):
    """

    :param one_square: one square of interest, used to calculate vectors
    :return: two vectors for the compensation to make
    """

    x_dist = one_square[0][0]-one_square[1][0]
    y_dist = one_square[0][1]-one_square[1][1]
    vector1 = np.array([x_dist, y_dist])
    vector1 = vector1/la.norm(vector1)

    vector2 = [vector1[1], -1*vector1[0]]

    return vector1, vector2


def rotate_coordinates_with_vectors(vector1, vector2, squares):

    Q = [[vector1[0], vector1[1]], [vector2[0], vector2[1]]]

    new_coords_dict = dict()

    for square in squares.keys():

        new_array = []
        for item in range(4):
            x = squares[square][item][0]
            y = squares[square][item][1]
            new_coord = np.dot(Q, [x, y])
            new_array.append(new_coord)

        new_coords_dict[square] = new_array

    return new_coords_dict


def extract_squares_from_gray_img(vector1, vector2, gray_img, new_coords):

    Q = [[vector1[0], vector1[1]], [vector2[0], vector2[1]]]
    Q_inv = la.inv(Q)

    new_imgs = dict()

    for square in new_coords.keys():
        coords = gdt.get_x_and_y_coords_for_plotting([square])
        x = coords[0]
        y = coords[1]

        max_x = int(math.ceil(max(x)))
        min_x = int(math.floor(min(x)))

        max_y = int(math.ceil(max(y)))
        min_y = int(math.floor(min(y)))

        new_matrix = np.zeros((max_x - min_x, max_y - min_y))

        for i in range(new_matrix.shape[0]):
            for j in range(new_matrix.shape[1]):
                old_coord = (np.dot([min_x+i, min_y+j], Q_inv))
                new_matrix[i,j] = gray_img[int(round(old_coord[1])),int(round(old_coord[0]))]

        new_imgs[square] = new_matrix

    return new_imgs
