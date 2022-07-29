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

    image_path = r"D:\LF171_Werker_Assistent_System\DATASET_VERSIONS\padded_256_256\images\New folder\\"
    mask_path = r"D:\LF171_Werker_Assistent_System\DATASET_VERSIONS\padded_256_256\masks\New folder\\"
    image_save_path = r"D:\LF171_Werker_Assistent_System\DATASET_VERSIONS\padded_256_256\images\New folder\\"
    mask_save_path =  r"D:\LF171_Werker_Assistent_System\DATASET_VERSIONS\padded_256_256\masks\New folder\\"
    image_paths = glob.glob(image_path + "*.jpg")
    mask_paths = glob.glob(mask_path + "*.png")
    PH1 = 23
    PH2 = 23
    PW1 = 55
    PW2 = 54

    for image_path, mask_path in zip(image_paths, mask_paths):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path,0)
        image = np.pad(image,((PH1,PH2), (PW1,PW2), (0,0)),"constant",constant_values=(0))
        print(image.shape)
        mask = np.pad(mask,((PH1,PH2), (PW1,PW2)),"constant",constant_values=(0))
        print(mask.shape)
        cv2.imwrite(image_path, image)
        cv2.imwrite(mask_path, mask)
    print("__")