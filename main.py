import matplotlib.pyplot as plt
import PIL.Image
import seaborn as sn
import pandas as pd
import os
from tqdm import tqdm

green_red=True



motion_classes = {
    (255, 140, 0):"next_into_driving_cut",
    (255, 165, 0):"next_into_driving_safe",
    (255, 0, 0):"driving_lane_approaching_front",
    (240, 128, 128):"driving_lane_leaving_front",
    (255, 99, 71):"driving_lane_stable",
    (255, 255, 0):"next_static_parallel",
    (255, 215, 0):"next_static_passing",
    (218, 165, 32):"next_static_passed",
    (60, 179, 113):"ramp_merging",
    (128, 0, 128):"opposite_vehicle",
    (0, 0, 255):"crossing_vehicle",
    (0, 255, 255):"turning_away",
    # (0,0,0): "background"
}

list_of_motion_classes = list(motion_classes.values())
matrix = []
for c1 in range(len(list_of_motion_classes)):
    matrix.append([])
    for c2 in list_of_motion_classes:
        matrix[c1].append(0)


for fn in tqdm(os.listdir("data/CenterLineImages_w/")):
    filename = fn.split('_')[0]
    try:
        path_to_traj = "data/CenterLineImages_w/"+filename+"_w_centerline.png"
        path_to_truth = "data/LabelsAutomatedStep1/"+filename+"_g_add_filtered.png"
        path_to_prediction = "data/predict_line/height_32/batch_56_channels/6400/train/"+filename+"_m.png"
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

        # for x,y in coords:
        #     # if the current pixel is not black, color it red
        #     if pred.getpixel((x,y)) != (0,0,0):
        #         pred.putpixel((x,y), (255,0,0))
        coords = set(coords)
        # coords.remove((0,0,0))
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
            else:
                # print("Error for coord " + str(coord))
                pass


    except:
        # print("Skipped filename " + filename)
        pass

# filename = "approaching (4)"





# motion_classes = {
#     "next_into_driving_cut":(255, 140, 0),
#     "next_into_driving_safe":(255, 165, 0),
#     "driving_lane_approaching_front":(255, 0, 0),
#     "driving_lane_leaving_front":(240, 128, 128),
#     "driving_lane_stable":(255, 99, 71),
#     "next_static_parallel":(255, 255, 0),
#     "next_static_passing":(255, 215, 0),
#     "next_static_passed":(218, 165, 32),
#     "ramp_merging":(60, 179, 113),
#     "opposite_vehicle":(128, 0, 128),
#     "crossing_vehicle":(0, 0, 255),
#     "turning_away":(0, 255, 255),
#     "parked_vehicles_same_side":(218, 165, 32),
#     "parked_vehicles_opposite_side":(128, 0, 128)
# }


#find truth values that are green


true_list_of_motion_classes = list_of_motion_classes.copy()
for l in range(len(true_list_of_motion_classes)):
    true_list_of_motion_classes[l] += "_t"
df_cm = pd.DataFrame(matrix, true_list_of_motion_classes, list_of_motion_classes)

plt.figure(figsize=(10,7))
sn.set_theme(font_scale=.75) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 6}) # font size
plt.xlabel('Predictions')
plt.ylabel('Truth')

plt.show()
plt.savefig('heatmap.png')
