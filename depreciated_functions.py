

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
