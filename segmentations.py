import numpy as np
import cv2

def binary(image, threshold=127, max_value=255, invert=False):
    """binary

    Args:
        image (cv): the image
        threshold (int, optional): threshold to apply. Defaults to 127.
        max_value (int, optional): max pixel value. Defaults to 255.
        invert (bool, optional): wether to use flag cv2.THRESH_BINARY_INV. Defaults to False.

    Returns:
        segmented_image
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, segmented_image = cv2.threshold(gray_image, threshold, max_value, cv2.THRESH_BINARY if invert == False else cv2.THRESH_BINARY_INV)

    return segmented_image

def otsu(image, invert=False):
    """otsu

    Args:
        image (cv): the image
        threshold (int, optional): threshold to apply. Defaults to 0.
        max_value (int, optional): max pixel value. Defaults to 255.

    Returns:
        segmented_image
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, segmented_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU if invert == False else cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    return segmented_image

def watershed(image, kernel_shape=(3,3), iterations=3, dist_ratio=0.3, return_borders=False):
    """Watershed
    Reference to https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html

    Args:
        image: the image
        kernel_shape (tuple, optional): the shape of the kernel to be used on morphological op. 
            Worth noting it is filled with 1's. Defaults to (3,3).
        iterations (int, optional): Number of iteration for dilation and open. Defaults to 3.
        dist_ratio (float, optional): Percentage of top X pixels furthest from background. Defaults to 0.3.
        return_borders (bool, optional): Wether to return binary or image with border. Defaults to False.

    Returns:
        image: either a binary image with each object or original image with border highlighted
    """
    binary_rep = binary(image, 127, invert=True)
    kernel = np.ones(kernel_shape, np.uint8)

    # Open than dilate image
    opened = cv2.morphologyEx(binary_rep, cv2.MORPH_OPEN, kernel, iterations=iterations)
    dilated = cv2.morphologyEx(opened, cv2.MORPH_DILATE, kernel, iterations=iterations)

    # gets distance from closest background pixel, than selects the 30% furthest
    #   the aim here is to get the thicker objects, or sure foreground (sure_fg)
    dist_transform = (cv2.distanceTransform(opened, cv2.DIST_L2, 5))
    _, sure_fg = cv2.threshold(dist_transform, (1 - dist_ratio) * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # gets pixels which are neither background nor in sure foreground
    in_between = cv2.subtract(dilated,sure_fg) 

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[in_between==255] = 0
    markers = (cv2.watershed(image, markers))

    if return_borders == True:
        image[markers == -1] = [255,0,0]
    else:
        markers[markers == -1] = 0
        markers[markers == 1] = 0
        image = np.uint8(np.interp(markers, (markers.min(), markers.max()), (0, 255)))

    return image
