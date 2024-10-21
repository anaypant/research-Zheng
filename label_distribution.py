import matplotlib.pyplot as plt
from constants import *
import time
import os
import PIL
from tqdm import tqdm

# file to create a distribution plot of the labels
# how many of each label is there in the dataset

labels=motion_classes.copy()
for label in labels:
    labels[label] = 0


# Execute for all files
directories = ["four_seconds", "one_second", "two_seconds"]

#Execute for one file
if TEST:
    directories = ["test_seconds"]

paths = ["test", "train"]

current_date = time.strftime("%Y_%m_%d_%H_%M_%S")
fn_set = set()

# count all the 'valid' files from directories
for directory in directories:
    for path in paths:
        for fn in os.listdir("data/truths/"+path+"/"):
            filename = fn.split('_')[0]
            # check that truths, preds and trajectories are all valid
            if os.path.exists("data/truths/"+path+"/"+filename+"_m.png") and os.path.exists("data/preds/"+directory+"/"+path+"/"+filename+"_m.png") and os.path.exists("data/trajectories/"+filename+"_w_centerline.png") and filename not in fn_set:
                fn_set.add(filename)

if label_count_trajectory:
    print("Counting labels from trajectories")
    # dictionary with coords for each file in truths
    # key: filename, value: list of coords
    # only need to do one time
    coords_dict = {}
    for filename in tqdm(fn_set):
        try:
            path_to_traj = "data/trajectories/"+filename+"_w_centerline.png"
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
            coords_dict[filename] = coords
        except:
            # print("Error in: " + fn)
            continue
    
    print("Done")

for directory in directories:
    for path in paths:
        print(directory + " " + path)
        for filename in fn_set:
            try:
                path_to_truth = "data/truths/"+path+"/"+filename+"_m.png"
                path_to_traj = "data/trajectories/"+filename+"_w_centerline.png"
                truth = PIL.Image.open(path_to_truth)

                if not label_count_trajectory:
                    for i in range(truth.size[0]):
                        for j in range(truth.size[1]):
                            if truth.getpixel((i,j)) in labels:
                                labels[truth.getpixel((i,j))] += 1
                
                else:
                    coords = coords_dict[filename]
                    for x,y in coords:
                        if truth.getpixel((x,y)) in labels:
                            labels[truth.getpixel((x,y))] += 1
                

            except:
                # print("Error in: " + fn)
                continue
        print(labels)
        # bar plot, make the tick labels vertical
        plt.figure(figsize=(12, 10))
        plt.xticks(rotation=90)
        plt.bar([motion_classes[label] for label in labels.keys()], labels.values(), color=[label_colors[motion_classes[label]] for label in labels.keys()])
        plt.title("Label Distribution")
        plt.xlabel("Labels")
        plt.ylabel("Frequency")
        # make sure none of the tick labels are cut off
        plt.tight_layout()
        

        # title
        # if trajectory based, add 'trajectory-based'
        # else, add 'area-based' to the title
        # add the directory and type to the title
        title = "Label Distribution"
        if label_count_trajectory:
            title += " (trajectory-based)"
        else:
            title += " (area-based)"
        title += " for " + directory + " " + path
        plt.title(title)

        # store it in a folder named timestamp and then label the file as the directory and path
        # if os path doesnt exist, create it
        if not os.path.exists("label_distribution"):
            os.makedirs("label_distribution")
        
        # save it in subfolder with timestamp
        if not os.path.exists("label_distribution/"+current_date):
            os.makedirs("label_distribution/"+current_date)
        plt.savefig("label_distribution/"+current_date+"/"+directory+"_"+path+".png")
        plt.close()

        # save the counts to a txt file in the same folder
        with open("label_distribution/"+current_date+"/"+directory+"_"+path+".txt", "w") as f:
            for label in labels:
                f.write(f"{motion_classes[label]}: {labels[label]}\n")
        