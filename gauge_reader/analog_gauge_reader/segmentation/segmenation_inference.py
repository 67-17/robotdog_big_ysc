from ultralytics import YOLO
import numpy as np
import cv2
from scipy import odr

MODEL_CACHE = {}


def segment_gauge_needle(image, model_path='best.pt'):
    """
    uses fine-tuned yolo v8 to get mask of segmentation
    :param img: numpy image
    :param model_path: path to yolov8 detection model
    :return: segmentation of needle
    """
    if model_path not in MODEL_CACHE:
        MODEL_CACHE[model_path] = YOLO(model_path)
    model = MODEL_CACHE[model_path]

    results = model.predict(
        image)  # run inference, detects gauge face and needle

    if len(results) == 0 or results[0].masks is None:
        raise Exception("No needle mask detected")
    mask_data = results[0].masks.data
    if mask_data is None or len(mask_data) == 0:
        raise Exception("No needle mask detected")
    first_mask = mask_data[0]
    if hasattr(first_mask, "cpu"):
        needle_mask = first_mask.cpu().numpy()
    else:
        needle_mask = first_mask.numpy()
    needle_mask_resized = cv2.resize(needle_mask,
                                     dsize=(image.shape[1], image.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

    y_coords, x_coords = np.where(needle_mask_resized)

    return x_coords, y_coords


def get_fitted_line(x_coords, y_coords):
    """
    Do orthogonal distance regression (odr) for this.
    """
    odr_model = odr.Model(linear)
    data = odr.Data(x_coords, y_coords)
    ordinal_distance_reg = odr.ODR(data, odr_model, beta0=[0.2, 1.], maxit=600)
    out = ordinal_distance_reg.run()
    line_coeffs = out.beta
    residual_variance = out.res_var
    return line_coeffs, residual_variance


def linear(B, x):
    return B[0] * x + B[1]


def get_start_end_line(needle_mask):
    return np.min(needle_mask), np.max(needle_mask)


def cut_off_line(x, y_min, y_max, line_coeffs):
    line = np.poly1d(line_coeffs)
    y = line(x)
    _cut_off(x, y, y_min, y_max, line_coeffs, 0)
    _cut_off(x, y, y_min, y_max, line_coeffs, 1)
    return x[0], x[1]


def _cut_off(x, y, y_min, y_max, line_coeffs, i):
    if y[i] > y_max:
        y[i] = y_max
        x[i] = 1 / line_coeffs[0] * (y_max - line_coeffs[1])
    if y[i] < y_min:
        y[i] = y_min
        x[i] = 1 / line_coeffs[0] * (y_min - line_coeffs[1])
