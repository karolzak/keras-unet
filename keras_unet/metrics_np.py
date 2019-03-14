import numpy as np

    
def iou_np(y_true, y_pred, smooth=1.):
    intersection = y_true * y_pred
    union = y_true + y_pred
    return np.sum(intersection + smooth) / np.sum(union - intersection + smooth)


def iou_thresholded_np(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred_pos = (y_pred > threshold) * 1.0
    intersection = y_true * y_pred_pos
    union = y_true + y_pred_pos
    return np.sum(intersection + smooth) / np.sum(union - intersection + smooth)


def iou_thresholded_np_imgwise(y_true, y_pred, threshold=0.5, smooth=1.):
    y_true = y_true.reshape((y_true.shape[0], y_true.shape[1]**2))
    y_pred = y_pred.reshape((y_pred.shape[0], y_pred.shape[1]**2))
    y_pred_pos = (y_pred > threshold) * 1.0
    intersection = y_true * y_pred_pos   
    union = y_true + y_pred_pos
    return np.sum(intersection + smooth, axis=1) / np.sum(union - intersection + smooth, axis=1)
