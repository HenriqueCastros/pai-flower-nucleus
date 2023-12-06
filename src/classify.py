import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from .description import *
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import LabelEncoder


def get_padding(image, pad_to):
    w, h = image.size
    h_padding = (pad_to - w) / 2
    v_padding = (pad_to - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    padding = [[int(t_pad), int(b_pad)], [int(l_pad), int(r_pad)], [0, 0]]
    return padding


def pad(img, pad_to=100):
    return tf.pad(
        tf.keras.utils.img_to_array(img, dtype=int),
        get_padding(img, pad_to),
        "CONSTANT",
        0,
    )


def classify_mahalanobis(
    img, path_to_model='./model_data/mahalanobis', subpath='multiclass'
):
    model = os.path.join(path_to_model, subpath)
    if os.path.isdir(model) == False:
        raise "The path to the model doesn't exist"

    with open(os.path.join(model, "class_means.json"), "r") as f:
        class_means = json.load(f)
        class_means = [pd.Series(series_dict) for series_dict in class_means]

    with open(os.path.join(model, "labels.json"), "r") as f:
        unique_classes = json.load(f)

    with open(os.path.join(model, "inv_cov_matrices.json"), "r") as f:
        inv_cov_matrices = json.load(f)

    data = (
        calculate_area(img),
        calculate_compactness(img),
        calculate_eccentricity(img),
        calculate_perimeter(img),
    )

    vals = {}
    for i, class_label in enumerate(unique_classes):
        vals[class_label] = mahalanobis(data, class_means[i], inv_cov_matrices[i])

    selected_class = min(vals, key=vals.get)

    return vals, selected_class


def classify_nn(
    img,
    path_to_model='./model_data/nn',
    model_type='baseline',
    subpath='multiclass',
    label_encoder_path='./model_data/nn/label_encoder_classes.npy',
):
    model = os.path.join(path_to_model, model_type, subpath)
    if os.path.isdir(model) == False:
        raise "The path to the model doesn't exist"

    model = tf.keras.models.load_model(os.path.join(model, 'model.keras'))

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_encoder_path)

    resize_ratio = 100 / max(img.size)
    img = img.resize([int(s * resize_ratio) for s in img.size])

    img = pad(img, max(*img.size, 100))

    prediction = model.predict(np.array([img]))

    vals = {}
    if prediction.shape[1] == 1:
        for i, c in enumerate(['Negative', 'Positive']):
            if i == 0:
                vals[c] = 1 - prediction[0, 0]
            elif i == 1:
                vals[c] = prediction[0, 0]
    else:
        for c, p in zip(label_encoder.classes_, prediction[0, :]):
            vals[c] = p

    selected_class = max(vals, key=vals.get)

    return vals, selected_class
