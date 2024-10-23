import matplotlib.pyplot as plt
from constants import *
import time
import os
import PIL
from tqdm import tqdm

# file to create a distribution plot of the labels
# how many of each label is there in the dataset


def exec():
    labels=motion_classes.copy()
    for label in labels:
        labels[label] = 0

    # Execute for all files
    paths = ["test", "train"]
    # switch between area-based and trajectory-based
    # area-based: count labels from the area of the image
    # trajectory-based: count labels from the trajectory image



    for label_count_trajectory in [False, True]:
        # set the label count type
        if label_count_trajectory:
            print("Counting labels from trajectories")
        else:
            print("Counting labels from areas")

        # create a timestamp for the folder

        current_date = time.strftime("%Y_%m_%d_%H_%M_%S")
        fn_set = set()


        # count all the 'valid' files from directories
        for path in paths:
            for fn in os.listdir("data/truths/"+path+"/"):
                filename = fn.split('_')[0]
                # check that truths, preds and trajectories are all valid
                if os.path.exists("data/truths/"+path+"/"+filename+"_m.png")  and os.path.exists("data/trajectories/"+filename+"_w_centerline.png") and filename not in fn_set:
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
            
        for path in paths:

            print(path)
            for label in labels:
                labels[label] = 0

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
            plt.title(title)

            # store it in a folder named timestamp and then label the file as the directory and path
            # if os path doesnt exist, create it
            if not os.path.exists("label_distribution"):
                os.makedirs("label_distribution")
            
            # save it in subfolder with timestamp
            if not os.path.exists("label_distribution/"+current_date):
                # create the folder based on the type
                if label_count_trajectory:
                    os.makedirs("label_distribution/"+current_date+"/trajectory_based")
                else:
                    os.makedirs("label_distribution/"+current_date+"/area_based")
            if label_count_trajectory:
                plt.savefig("label_distribution/"+current_date+"/trajectory_based/"+path+"_trajectory_based.png")
            else:
                plt.savefig("label_distribution/"+current_date+"/area_based/"+path+"_area_based.png")
            plt.close()

            # save the counts to a txt file in the same folder
            if label_count_trajectory:
                with open("label_distribution/"+current_date+"/trajectory_based/"+path+"_trajectory_based.txt", "w") as f:
                    # write the labels dictionary to the file nicely
                    for key in labels:
                        # write the motion class of the key
                        f.write(motion_classes[key] + ": " + str(labels[key]) + "\n")
            else:
                with open("label_distribution/"+current_date+"/area_based/"+path+"_area_based.txt", "w") as f:
                    for key in labels:
                        # write the motion class of the key
                        f.write(motion_classes[key] + ": " + str(labels[key]) + "\n")

if __name__ == "__main__":
    exec()