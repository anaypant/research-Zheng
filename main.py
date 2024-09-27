import matplotlib.pyplot as plt
import PIL.Image
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


green_red=True


def calculate_f1_iou(confusion_matrix, class_labels):
    f1_scores = {}
    iou_scores = {}
    
    for i, label in enumerate(class_labels):
        TP = confusion_matrix[i, i]  # True Positives
        FP = np.sum(confusion_matrix[:, i]) - TP  # False Positives
        FN = np.sum(confusion_matrix[i, :]) - TP  # False Negatives
        TN = np.sum(confusion_matrix) - (TP + FP + FN)  # True Negatives (optional if needed)

        # Precision, Recall, F1-score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # IoU Calculation
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        
        f1_scores[label] = f1
        iou_scores[label] = iou

    return f1_scores, iou_scores




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





# for testing purposes
# directories = ["one_second"]
# paths = ["test"]


#Execute for all files
directories = ["four_seconds", "one_second", "two_seconds"]
paths = ["test", "train"]


for directory in directories:
    for path in paths:
        list_of_motion_classes = list(motion_classes.values())
        matrix = []
        true_m = []
        pred_m = []
        for c1 in range(len(list_of_motion_classes)):
            matrix.append([])
            for c2 in list_of_motion_classes:
                matrix[c1].append(0)
        print(directory + " " + path)
        for fn in tqdm(os.listdir("data/preds/"+directory+"/"+path+"/")):
            filename = fn.split('_')[0]
            try:
                path_to_traj = "data/trajectories/"+filename+"_w_centerline.png"
                path_to_truth = "data/truths/"+filename+"_g_add_filtered.png"
                path_to_prediction = "data/preds/"+directory+"/"+path+"/"+filename+"_m.png"
                #get image data, view image with PIL
                truth = PIL.Image.open(path_to_truth)
                pred = PIL.Image.open(path_to_prediction)
                traj = PIL.Image.open(path_to_traj)
                # print("Found filename " + filename)


                coords = []
                if not green_red:
                    for i in range(traj.size[0]):
                        for j in range(traj.size[1]):
                            if traj.getpixel((i,j)) == (0, 255, 0):
                                coords.append((i,j))
                else:
                    # green and red pixels
                    for i in range(traj.size[0]):
                        for j in range(traj.size[1]):
                            if traj.getpixel((i,j)) == (0, 255, 0) or traj.getpixel((i,j)) == (255, 0, 0):
                                coords.append((i,j))

                #normalize all coords to the dimensions of the smaller image
                scale = (pred.size[0]/traj.size[0], pred.size[1]/traj.size[1])
                coords = [(int(scale[0]*x), int(scale[1]*y)) for x,y in coords]
                #map the coords to the pred image, remove duplicates
                coords = set(coords)
                coords = list(coords)




                for coord in coords:
                    t1 = truth.getpixel(coord)
                    t2 = pred.getpixel(coord)
                    if t1 in motion_classes and t2 in motion_classes:
                        c1 = motion_classes[truth.getpixel(coord)]
                        c2 = motion_classes[pred.getpixel(coord)]
                        keyloc1 = list_of_motion_classes.index(c1)
                        keyloc2 = list_of_motion_classes.index(c2)
                        matrix[keyloc1][keyloc2] += 1
                        true_m.append(c1)
                        pred_m.append(c2)
                    else:
                        # print("Error for coord " + str(coord))
                        pass


            except:
                # print("Skipped filename " + filename)
                pass

        true_list_of_motion_classes = list_of_motion_classes.copy()
        for l in range(len(true_list_of_motion_classes)):
            true_list_of_motion_classes[l] += "_t"
        df_cm = pd.DataFrame(matrix, true_list_of_motion_classes, list_of_motion_classes)

        cm = confusion_matrix(true_m, pred_m, labels=list_of_motion_classes)
        fig = plt.figure(figsize=(11, 11))
        gs = fig.add_gridspec(3, 3, width_ratios=[0.2, 3, 0.2], height_ratios=[0.2,3,0.2])

        ax_cm = fig.add_subplot(gs[1, 1])
        custom_cmap = plt.get_cmap('Reds')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list_of_motion_classes)
        disp.plot(ax=ax_cm, xticks_rotation='vertical', cmap=custom_cmap, colorbar=False)
        for text in ax_cm.texts:
                text.set_fontsize(7)

        cbar_ax = fig.add_subplot(gs[1, 2])
        fig.colorbar(disp.im_, cax=cbar_ax)

        bins = np.arange(len(list_of_motion_classes) + 1) - 0.5

        #histogram x
        ax_hist_x = fig.add_subplot(gs[2, 1], sharex=ax_cm)
        sns.kdeplot([list_of_motion_classes.index(x) for x in pred_m], bw_adjust=1.5, fill=True, ax=ax_hist_x, color='skyblue')
        ax_hist_x.axvline(np.mean([list_of_motion_classes.index(x) for x in pred_m]), color='red', linestyle='dashed', linewidth=2)
        ax_hist_x.set_ylabel('Density')
        ax_hist_x.grid(True)

        #histogram y
        ax_hist_y = fig.add_subplot(gs[1, 0], sharey=ax_cm)
        sns.kdeplot([list_of_motion_classes.index(x) for x in true_m], bw_adjust=1.5, fill=True, ax=ax_hist_y, color='skyblue', vertical=True)
        ax_hist_y.axhline(np.mean([list_of_motion_classes.index(x) for x in true_m]), color='red', linestyle='dashed', linewidth=2)
        ax_hist_y.set_xlabel('Density')
        ax_hist_y.grid(True)

        plt.setp(ax_hist_x.get_xticklabels(), visible=False)
        plt.setp(ax_hist_y.get_yticklabels(), visible=False)
        
        for label in ax_cm.get_xticklabels():
            label_text = label.get_text()
            label.set_color(label_colors.get(label_text, 'black'))  # Default to black if no color found

        # Apply custom colors to y-tick labels
        for label in ax_cm.get_yticklabels():
            label_text = label.get_text()
            label.set_color(label_colors.get(label_text, 'black'))  # Default to black if no color found

        f1_scores, iou_scores = calculate_f1_iou(cm, list_of_motion_classes)

        legend_text = []
        for label in list_of_motion_classes:
            f1 = f1_scores[label]
            iou = iou_scores[label]
            legend_text.append(f"{label}: F1 = {f1:.2f}, IoU = {iou:.2f}")

        # Create the legend
        # plt.legend(legend_text, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')


        # Adjust layout for better spacing
        plt.tight_layout()
        plt.suptitle("Conf Matrix for " + directory.split("_")[0] + " " + directory.split("_")[1] + " " + path)

        plt.savefig(directory+"_"+path+'.png')

