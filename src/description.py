import cv2
import numpy as np

def calculate_area(image):
    """calculate_area
        Considers any non-zero pixel as part of object. 0 is considered as background.

    Args:
        image: the image to calculate.  

    Returns:
        number: the area of the object
    """
    return np.count_nonzero(image)

def calculate_perimeter(image):
    """calculate_perimeter
        If the image has multiple objects, it will only return one perimeter
        TODO: return an array with all perimeters
    
    Args:
        image: the image to be calculated

    Returns:
        number: the perimeter
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.arcLength(contours[0], True)

def calculate_centroid(image):
    """calculate_centroid
    
    Args:
        image: the image to be calculated

    Returns:
        tuple: the centroid
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    M = cv2.moments(image)
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)

def calculate_compactness(image):
    """calculate_compactness

    Args:
        image: the image to be calculated

    Returns:
        number: compactness
    """
    return (calculate_perimeter(image) ** 2) / (4 * np.pi * calculate_area(image))

def calculate_eccentricity(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(contour)
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    eccentricity = np.sqrt(1 - (minor_axis * 2) / (major_axis * 2))
    return eccentricity