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

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def crop_image_around_point(image, x, y, crop_size):
    """crop_image_around_point
        Crops the image using (x,y) as center. Which means the position of cropped
    area is ((x - (crop_size / 2), (y - (crop_size / 2)) to 
    ((x + (crop_size / 2), (y + (crop_size / 2))
    
    Args:
        image (list[list]): image to be cropped
        x (int): x position
        y (int): y position
        crop_size (int): distance from center to be cropped

    Returns:
        list[list]: cropped image
    """
    x_start = max(0, x - int(crop_size / 2))
    y_start = max(0, y - int(crop_size / 2))
    x_end = min(image.shape[1], x + int(crop_size / 2))
    y_end = min(image.shape[0], y + int(crop_size / 2))

    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image

def getOriginFromCropped(x, y, crop_size):
    """getOriginFromCropped

    Args:
        x (int): x position
        y (int): y position
        crop_size (int): distance from center to be cropped

    Returns:
        tuple: origin point
    """
    x_start = max(0, x - int(crop_size / 2))
    y_start = max(0, y - int(crop_size / 2))
    return (x_start, y_start)