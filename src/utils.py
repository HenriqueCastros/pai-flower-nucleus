def scale_proportional(tuple_to_scale, max_dimensions=(400, 200)):
    """scale_proportional

    Args:
        tuple_to_scale (tuple): (x,y)
        max_dimensions (tuple, optional): max size for x, y. Defaults to (400, 200).

    Returns:
        (tuple): scaled tuple 
    """
    x, y = tuple_to_scale
    max_x, max_y = max_dimensions

    scale_x = min(1, max_x / x)
    scale_y = min(1, max_y / y)

    scale_factor = min(scale_x, scale_y)
    scaled_tuple = (int(x * scale_factor), int(y * scale_factor))

    return scaled_tuple