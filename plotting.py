import general as gen
from matplotlib import pyplot as plt

def plot_all_squares(array_of_square_coords):
    """
    :param array_of_square_coords: array of square coordinate arrays (from find_squares function)
    :return: Plot of all squares in 2D space
    """
    square_coords = []
    for square in array_of_square_coords:
        coords = gen.get_x_and_y_coords(square)
        plt.scatter(coords[0], coords[1])
        square_coords.append(coords)


def plot_from_dict_of_squares(dict_of_squares):
    """
    this is what my function does
    :param dict_of_squares:
    :return:
    """
    for square in dict_of_squares.keys():
        coords = gen.get_x_and_y_coords(dict_of_squares[square])
        plt.scatter(coords[0], coords[1])
