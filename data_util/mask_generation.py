import json, os, glob
from PIL import Image, ImageDraw
import numpy as np
import cv2 
import matplotlib.pyplot as plt

class_dict = {'Parts':1}
json_file = r'C:\Users\bakthavatchalam\Desktop\02_Work_assistance\negative\\annotations.json'
all_labels = json.load(open(json_file))
# rawfile = r'D:\01_Thesis\Mask_RCNN-master\samples\custom\dataset\train\1.jpeg'
# file_size = os.path.getsize(rawfile)
# print(file_size)
# rawfiledict = rawfile + str(file_size)
# rawfiledict = rawfiledict.split('\\')[-1]
# data = all_labels[rawfiledict]
# data
all_labels

save_path = r'C:\Users\bakthavatchalam\Desktop\02_Work_assistance\negative\masks\\'
image_save_path = r'C:\Users\bakthavatchalam\Desktop\02_Work_assistance\negative\images\\'
image_path = r'C:\Users\bakthavatchalam\Desktop\02_Work_assistance\negative\\'
lab_keys = all_labels.keys()

for k in lab_keys:
    data = all_labels[k]
    file_name = all_labels[k]["filename"]
    print(file_name)
    image = cv2.imread(image_path + file_name)
    file_name = str(int(file_name.split(".")[0]) + 28) + ".jpg"
    print(file_name)
    blank = np.zeros((210,147))
    
    cv2.imwrite(image_save_path + file_name,image)
    # for region in range(len(data["regions"])):
    #     print(region)
    
    # for region in range(len(data["regions"])):
    x = data["regions"]["0"]["shape_attributes"]["all_points_x"]
    y = data["regions"]["0"]["shape_attributes"]["all_points_y"]
    points = []

    shape = "Part"#data["regions"]['0']["region_attributes"]["names"]

    for idx in range(len(x)):
        temp = np.array([x[idx],y[idx]], np.int32)
        points.append(temp)

    points = np.array(points, np.int32)
    isClosed = True
    color = (255, 255, 255)
    fill = color #(200, 200, 200)
    # Blue color in BGR
    if shape == "Part":

        color = (125, 125, 125)
        fill = (50, 50, 50)


    # Line thickness of 2 px

    # Using cv2.polylines() method
    # Draw a Blue polygon with 
    # thickness of 1 px
    thickness = 2
    
    image = cv2.drawContours(blank, [points], -1, fill, -1)
    
    image = cv2.polylines(blank, [points], 
                            isClosed, color, thickness)
    
    cv2.imwrite(save_path + file_name,image)

print("--")