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
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    
    if(len(image.shape)<3):
        gray_image = image
    else:
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
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    
    if(len(image.shape)<3):
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, segmented_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU if invert == False else cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    return segmented_image

def watershed(image, kernel_shape=(3,3), iterations=3, dist_ratio=0.3, return_borders=False, bin_threshhold=127):
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
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    binary_rep = binary(image, bin_threshhold, invert=True)
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

def region_growing(image, seed=None, return_binary=True, threshold=[25,25,25]):
    """region_growing

    Args:
        image: image to be segmented
        seed (tuple, optional): seed from which growing will start. Defaults to None.
        return_binary (bool, optional): if false return the actual pixel of the image instead of a white. Defaults to True.
        threshold (list, optional): thresh array of max absolute difference in each colour spectrum. Defaults to [20,20,20].

    Returns:
        np.ndarray: image segemented
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 4-connectivity

    height, width, _ = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)

    if seed == None:
        seed = (height//2, width//2)

    queue = [seed]
    seed_color = image[seed[0], seed[1]]

    while queue:
        current_pixel = queue.pop(0)
        x, y = current_pixel
        if 0 <= x < height and 0 <= y < width:
            if np.all(segmented[x, y] == 0) and np.all(np.abs(image[x, y].astype(np.int64) - seed_color.astype(np.int64)) < threshold):
                if return_binary == True:
                    segmented[x, y] = (255,255,255)
                else:
                    segmented[x, y] = image[x, y]
                for neighbor in neighbors:
                    queue.append((x + neighbor[0], y + neighbor[1]))
    
    return segmented