import os
import json
import pandas as pd
from .description import *
from scipy.spatial.distance import mahalanobis

def classify_mahalanobis(img, path_to_model='./model_data/mahalanobis/'):
    if os.path.isdir(path_to_model) == False:
        raise "The path to the model doesn't exist"

    with open(path_to_model + "class_means.json", "r") as f:
        class_means = json.load(f)
        class_means = [pd.Series(series_dict) for series_dict in class_means]

    with open(path_to_model+"labels.json", "r") as f:
        unique_classes = json.load(f)

    with open(path_to_model+"inv_cov_matrices.json", "r") as f:
        inv_cov_matrices = json.load(f) 
    
    data = (
        calculate_area(img),
        calculate_compactness(img),
        calculate_eccentricity(img),
        calculate_perimeter(img)
    )

    arr = np.zeros((1, len(unique_classes))).astype(object)
    for i, class_label in enumerate(unique_classes):
        arr[:, i] = {class_label: mahalanobis(data, class_means[i], inv_cov_matrices[i])}
    
    return arr