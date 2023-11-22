import cv2

def binary_segmentation(image, threshold=127, max_value=255, invert=False):
    """binary_segmentation

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

def otsu_segmentation(image, invert=False):
    """otsu_segmentation

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
