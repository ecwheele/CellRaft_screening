import detect_grids_on_array as gdt


def extract_array_from_image(square, image):
    """
    gets 3D array of pixel from a image given square coordinates
    :param square: square to extract
    :param image: image to extract from
    :return: 3D array of pixel intensities from the square
    """
    x_and_y = gdt.get_x_and_y_coords_for_plotting(square)

    min_x = min(x_and_y[0])
    max_x = max(x_and_y[0])
    min_y = min(x_and_y[1])
    max_y = max(x_and_y[1])

    img_array = image[min_y:max_y, min_x:max_x]

    return img_array


