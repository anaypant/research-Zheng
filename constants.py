import numpy as np


RED_GREEN = True

# Possible maplotlib color maps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
COLOR = "Reds"
TEST = False

motion_classes = {
    (255, 140, 0):"fast cut in",
    (255, 165, 0):"slow cut in",
    (255, 0, 0):"same close front",
    (240, 128, 128):"same leave",
    (255, 99, 71):"same stable",
    (255, 255, 0):"next parallel",
    (255, 215, 0):"next passing",
    (218, 165, 32):"next passed",
    (60, 179, 113):"ramp",
    (128, 0, 128):"opposite",
    (0, 0, 255):"crossing",
    (0, 255, 255):"turning",
    # (0,0,0): "background"
}
label_colors = {
    "fast cut in": "orange",        # (255, 140, 0)
    "slow cut in": "darkorange",    # (255, 165, 0)
    "same close front": "red",      # (255, 0, 0)
    "same leave": "lightcoral",     # (240, 128, 128)
    "same stable": "tomato",        # (255, 99, 71)
    "next parallel": "yellow",      # (255, 255, 0)
    "next passing": "gold",         # (255, 215, 0)
    "next passed": "goldenrod",     # (218, 165, 32)
    "ramp": "mediumseagreen",       # (60, 179, 113)
    "opposite": "purple",           # (128, 0, 128)
    "crossing": "blue",             # (0, 0, 255)
    "turning": "cyan"               # (0, 255, 255)
}


def calculate_f1_iou(confusion_matrix, class_labels):
    f1_scores = {}
    iou_scores = {}
    
    for i, label in enumerate(class_labels):
        TP = confusion_matrix[i, i]  # True Positives
        FP = np.sum(confusion_matrix[:, i]) - TP  # False Positives
        FN = np.sum(confusion_matrix[i, :]) - TP  # False Negatives
        TN = np.sum(confusion_matrix) - (TP + FP + FN)  # True Negatives

        # Precision, Recall, F1-score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # IoU Calculation
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        
        f1_scores[label] = f1
        iou_scores[label] = iou

    return f1_scores, iou_scores


