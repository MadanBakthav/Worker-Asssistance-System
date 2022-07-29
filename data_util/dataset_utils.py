import cv2
import numpy as np
import glob
import os

def crop_and_save(image, bbox, file_name):
    cropped_image = image[bbox[0]: bbox[0] + bbox[2], bbox[1]: bbox[1] + bbox[3], :]
    cv2.imwrite(file_name, cropped_image)
    
def generate_negative_description_file(text_file_name, folder_path):
    # open the output file for writing. will overwrite all existing data in there
    with open(text_file_name, 'w') as f:
        # loop over all the filenames
        for filename in os.listdir(folder_path):
            f.write( folder_path + '/' + filename + '\n')
            
if __name__ == "__main__":
    
    ####################################################
    ###     Cropping the images                      ###
    ####################################################

    bbox = (543, 549, 105, 106) # y x w h from paint
    images_path = glob.glob(r"D:\LF171_Werker_Assistent_System\Scripts\inspection\\*.jpg")
    save_path = r"D:\LF171_Werker_Assistent_System\Scripts\inspection\\"

    bbox = (530, 698, 106, 105)
    images_path = glob.glob(r"D:\LF171_Werker_Assistent_System\DATASET_VERSIONS\inspection\\*.jpg")
    print(images_path)
    save_path = r"D:\LF171_Werker_Assistent_System\DATASET_VERSIONS\inspection\\cropped\\"

    for i, path in enumerate(images_path):
        image = cv2.imread(path)
        file_name = save_path + str(i) + ".jpg"
        crop_and_save(image, bbox, file_name)
        
    ####################################################
    ###     Generating Description file              ###
    ####################################################
    # generate_negative_description_file("neg.txt", "negative")

    
    print("---")