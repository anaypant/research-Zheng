import matplotlib.pyplot as plt
import PIL.Image
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import time
from constants import *


# Execute for all files
directories = ["four_seconds", "one_second", "two_seconds"]

#Execute for one file
if TEST:
    directories = ["test_seconds"]

paths = ["test", "train"]

current_date = time.strftime("%Y_%m_%d_%H_%M_%S")




for directory in directories:
    for path in paths:
        list_of_motion_classes = list(motion_classes.values())
        true_m = []
        pred_m = []
        print(directory + " " + path)
        for fn in tqdm(os.listdir("data/preds/"+directory+"/"+path+"/")):
            filename = fn.split('_')[0]
            try:
                path_to_traj = "data/trajectories/"+filename+"_w_centerline.png"
                path_to_truth = "data/truths/"+path+"/"+filename+"_m.png"
                path_to_prediction = "data/preds/"+directory+"/"+path+"/"+filename+"_m.png"
                
                #get image data, view image with PIL
                truth = PIL.Image.open(path_to_truth)
                pred = PIL.Image.open(path_to_prediction)
                traj = PIL.Image.open(path_to_traj)

                coords = []
                if not RED_GREEN:
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
                        true_m.append(c1)
                        pred_m.append(c2)
                    else:
                        pass
            except:
                pass
        
        # Creating a confusion matrix from the truths and predictions
        cm = confusion_matrix(true_m, pred_m, labels=list_of_motion_classes)

        # Creating matplotlib plot to plot everything
        fig_size = 12
        fig = plt.figure(figsize=(fig_size, fig_size))
        gs = fig.add_gridspec(3, 3, width_ratios=[2, 12, 2], height_ratios=[2,12,2])
        ax_cm = fig.add_subplot(gs[1, 1])
        custom_cmap = plt.get_cmap(COLOR)

        # Confusion Matrix Display - sklearn
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list_of_motion_classes)
        disp.plot(ax=ax_cm, xticks_rotation='vertical', cmap=custom_cmap, colorbar=False)
        
        # sklearn color bar
        cbar_ax = fig.add_subplot(gs[1, 2])
        fig.colorbar(disp.im_, cax=cbar_ax)

        # seaborne kdeplot pred (x axis)
        ax_hist_x = fig.add_subplot(gs[2, 1])
        sns.kdeplot([list_of_motion_classes.index(x) for x in pred_m], bw_adjust=1.5, fill=True, ax=ax_hist_x, color='skyblue')
        ax_hist_x.axvline(np.mean([list_of_motion_classes.index(x) for x in pred_m]), color='red', linestyle='dashed', linewidth=2)
        ax_hist_x.set_ylabel('Density')
        ax_hist_x.set_yticklabels([])
        
        # seaborne kdeplot truth (y axis)
        ax_hist_y = fig.add_subplot(gs[1, 0])
        sns.kdeplot([list_of_motion_classes.index(x) for x in true_m], bw_adjust=1.5, fill=True, ax=ax_hist_y, color='skyblue', vertical=True, clip=None)
        ax_hist_y.axhline(np.mean([list_of_motion_classes.index(x) for x in true_m]), color='red', linestyle='dashed', linewidth=2)
        ax_hist_y.set_xlabel('Density')
        ax_hist_y.set_xticklabels([])  # This removes the vertical axis labels


        # Set colors of confusion matrix x-axis
        for label in ax_cm.get_xticklabels():
            label_text = label.get_text()
            label.set_color(label_colors.get(label_text, 'black'))  # Default to black if no color found

        # Apply custom colors to y-tick labels
        for label in ax_cm.get_yticklabels():
            label_text = label.get_text()
            label.set_color(label_colors.get(label_text, 'black'))  # Default to black if no color found


        # f1 and iou scores
        f1_scores, iou_scores = calculate_f1_iou(cm, list_of_motion_classes)

        output_filename = f"results/{current_date}/{directory}_{path}.txt"
        # if folder does not exist, create it
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))


        with open(output_filename, "w") as file:
            file.write("F1 and IoU Scores for " + directory + " " + path + "\n\n")
            
            file.write("F1 Scores:\n")
            for label, score in f1_scores.items():
                file.write(f"{label}: {score:.4f}\n")
            
            file.write("\nIoU Scores:\n")
            for label, score in iou_scores.items():
                file.write(f"{label}: {score:.4f}\n")


        plt.tight_layout()
        plt.suptitle("Conf Matrix for " + directory.split("_")[0] + " " + directory.split("_")[1] + " " + path)

        plt.savefig("results/"+current_date+"/"+directory+"_"+path+'.png')
        plt.close()
        