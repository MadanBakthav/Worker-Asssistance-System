from matplotlib.contour import ContourSet
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import threading
import matplotlib.pylab as plt
from skimage.metrics import structural_similarity



def similarity(ref_image_path, current_image):
    """ similarity is the function used to check if the part is assembled correctly or not.
    It takes the exact pixels where the fixture is located in the image. The reference image is provided 
    for comparing the assembly. It returns a similarity score based on the difference between the actual
    scene and the reference image.

    Args:
        ref_image_path (np.array): Image took for comparing the assembly steps. It is cropped to ROI(to the fixture).
        current_image (np.array): Image cropped to the ROI from the camera feed.

    Returns:
        similarity_score(float): It is the measure of similarity between the given images. It ranges between 0 to 1. 1 being the 
        exactly similary and 0 being not even a single pixel is similar between the given images.
    """
    ref_image_path = ref_image_path
    after = current_image 
    before = cv2.imread(ref_image_path)
    after = cv2.GaussianBlur(after,(3,3),cv2.BORDER_DEFAULT)
    before = cv2.GaussianBlur(before,(3,3),cv2.BORDER_DEFAULT)
    # converting to grayscale image
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    (similarity_score, diff) = structural_similarity(before_gray, after_gray, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]
    # print("SIMILARITY SCORE : ", similarity_score)

    return similarity_score


def disassembly_op1_similarity(ref_image_path, current_image):
    """disassembly_op1_similarity is the function used to check if the part is assembled correctly or not.
    It is similar to the similarity function. The difference is, this is developed to check only if the lid placed on 
    the pi  is properly assembled or not. This is done by slicing the given image more precise to the lid region.
    This improved the inspection process's stability.

    Args:
        ref_image_path (np.array): Image took for comparing the assembly steps. It is cropped to ROI(to the fixture).
        current_image (np.array): Image cropped to the ROI from the camera feed.

    Returns:
        similarity_score(float): It is the measure of similarity between the given images. It ranges between 0 to 1. 1 being the 
        exactly similary and 0 being not even a single pixel is similar between the given images.
    """
    ref_image_path = ref_image_path
    after = current_image [27:87,12:66,:]
    before = cv2.imread(ref_image_path)[27:87,12:66,:]
    after = cv2.GaussianBlur(after,(3,3),cv2.BORDER_DEFAULT)
    before = cv2.GaussianBlur(before,(3,3),cv2.BORDER_DEFAULT)

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    (similarity_score, diff) = structural_similarity(before_gray, after_gray, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]

    #print("DIS 1 ",similarity_score)
    return similarity_score#, contours


def Inspect (image, frame,  list_text_dict, assembly_stage, assembly_flag):
    """Inspection function monitors the entire inspection process. It checks for similarity between the images based on the threshold.
    It takes the current image, list_text_dict (carries the information on the progress of process, description for display and finally the 
    path to display the reference image), assembly_stage and the assembly flag(needs to be removed).

    Args:
        image (np.array): frame sliced to ROI
        frame (np.array): Current frame from the camerafeed
        list_text_dict (dict): dictionary containing the progress of assembly process, description of process and reference image path for display. 
        assembly_stage (int): progress of assembly process
        assembly_flag (bool): Not currenly in use. This can be removed

    Returns: [list_text_dict["assembly_stage"], list_text_dict["description"], list_text_dict["ref_image"]], assembly_stage, assembly_flag, list_text_dict
        new_frame (np.array): The frame with highlights with respect to the assembly process.
        list: Containing the similary information as list_text_dict
        assembly_stage (int): progress of assembly process
        assembly_flag (bool): Not currenly in use. This can be removed
    """
    alpha = 0.5
    list_text_dict = list_text_dict
    ins_threshold = 0.82
    ins_threshold_op_4 = 0.905
    new_frame = frame

    assembly_stage = assembly_stage
    print(assembly_stage)
    
    if similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_1.jpg', image) >= ins_threshold:
        overlay = frame.copy()
        x, y, w, h =  525, 60, 200, 300  # Rectangle parameters
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1)
        p_x, p_y, p_w, p_h = 565, 565, 75, 60
        cv2.rectangle(overlay, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 200, 0), -1)
        new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        new_frame = cv2.arrowedLine(new_frame, (int(x + 0.5*w), y + h), (600, 565),
                                    (0,255,255), 3)
        assembly_stage = 1
        assembly_flag = False
        list_text_dict["assembly_stage"] = assembly_stage
        list_text_dict["description"] = "Please take the Part 1 and place it in the fixture as shown"
        list_text_dict["ref_image"] = r"C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\ref_2.png"

    elif disassembly_op1_similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_3.jpg', image) >= ins_threshold:
        overlay = frame.copy()
        x, y, w, h =  260, 60, 200, 300  # Rectangle parameters
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1) 
        p_x, p_y, p_w, p_h = 569, 569, 45, 60
        cv2.rectangle(overlay, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 200, 0), -1)
        # print(Message)
        new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        new_frame = cv2.arrowedLine(new_frame, (int(x + 0.5*w), y + h), (600, 565),
                                    (0,255,255), 3)
        assembly_stage = 3
        assembly_flag = False
        list_text_dict["assembly_stage"] = assembly_stage
        list_text_dict["description"] = "Please take the Part 3 and assemble it as shown"
        list_text_dict["ref_image"] = r"C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\ref_4.png"
    
    elif similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_2.jpg', image) >= ins_threshold:
        overlay = frame.copy()
        x, y, w, h =  800, 60, 200, 300  # Rectangle parameters
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1) 
        p_x, p_y, p_w, p_h = 569, 569, 45, 60
        cv2.rectangle(overlay, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 200, 0), -1)
        # print(Message)
        new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        new_frame = cv2.arrowedLine(new_frame, (int(x + 0.5*w), y + h), (600, 565),
                                    (0,255,255), 3)
        assembly_stage = 2
        assembly_flag = False
        list_text_dict["assembly_stage"] = assembly_stage
        list_text_dict["description"] = "Please take the Pi and place it on the Part 1 as shown"
        list_text_dict["ref_image"] = r"C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\ref_3.png"



    elif similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_4.jpg', image) >= ins_threshold_op_4:
        overlay = frame.copy()

        assembly_stage = 4
        assembly_flag = False
        list_text_dict["assembly_stage"] = assembly_stage
        list_text_dict["description"] = "Please place the assembly carefully in the bin."
        list_text_dict["ref_image"] = r"C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\ref_1.png"
    
    elif (similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_4.jpg', image) < ins_threshold_op_4 and
        similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_3.jpg', image) < ins_threshold and
        similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_2.jpg', image) < ins_threshold and 
        similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_1.jpg', image)):
        
        if assembly_flag == False and assembly_stage > 1:
            assembly_stage = max(0, assembly_stage)
            assembly_flag = True
            list_text_dict["assembly_stage"] = assembly_stage
            list_text_dict["description"] = " Check if the part is correctly assembled as shown or Move your hands away for inspection."
            list_text_dict["ref_image"] = list_text_dict["ref_image"]
        else:
            assembly_stage = max(0, assembly_stage)
            assembly_flag = True
            list_text_dict["assembly_stage"] = assembly_stage
            list_text_dict["description"] = "Check if the part is correctly assembled as shown or Move your hands away for inspection."
            list_text_dict["ref_image"] = list_text_dict["ref_image"]
        return new_frame, [list_text_dict["assembly_stage"], list_text_dict["description"], list_text_dict["ref_image"]], assembly_stage, assembly_flag, list_text_dict
    
    return new_frame, [list_text_dict["assembly_stage"], list_text_dict["description"], list_text_dict["ref_image"]], assembly_stage, assembly_flag, list_text_dict




def Disassemble_Inspect (image, frame,  list_text_dict, assembly_stage, assembly_flag):
    """Disassemble_Inspect function monitors the entire inspection process for disassembling process. It checks for similarity between the images based on the threshold.
    It takes the current image, list_text_dict (carries the information on the progress of process, description for display and finally the 
    path to display the reference image), assembly_stage and the assembly flag(needs to be removed).

    Args:
        image (np.array): frame sliced to ROI
        frame (np.array): Current frame from the camerafeed
        list_text_dict (dict): dictionary containing the progress of assembly process, description of process and reference image path for display. 
        assembly_stage (int): progress of assembly process
        assembly_flag (bool): Not currenly in use. This can be removed

    Returns: [list_text_dict["assembly_stage"], list_text_dict["description"], list_text_dict["ref_image"]], assembly_stage, assembly_flag, list_text_dict
        new_frame (np.array): The frame with highlights with respect to the disassembly process.
        list: Containing the similary information as list_text_dict
        assembly_stage (int): progress of disassembly process
        assembly_flag (bool): Not currenly in use. This can be removed
    """
    alpha = 0.5
    list_text_dict = list_text_dict
    ins_threshold = 0.8
    ins_threshold_op_4 = 0.75
    new_frame = frame

    assembly_stage = assembly_stage
    print(assembly_stage)


    if disassembly_op1_similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_4.jpg', image) >= ins_threshold_op_4:
        overlay = frame.copy()

        assembly_stage = 1
        assembly_flag = False
        list_text_dict["assembly_stage"] = assembly_stage
        list_text_dict["description"] = "Place the Part 1 carefully in the bin"
        list_text_dict["ref_image"] = r"C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\ref_3.png"
        x, y, w, h =  260, 60, 200, 300  # Rectangle parameters
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1) 
        p_x, p_y, p_w, p_h = 569, 569, 45, 60
        cv2.rectangle(overlay, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 200, 0), -1)
        # print(Message)
        new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        new_frame = cv2.arrowedLine(new_frame,  (600, 565),(int(x + 0.5*w), y + h),
                                    (0,255,255), 3)
    
    elif similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_1.jpg', image) >= ins_threshold:
        overlay = frame.copy()

        assembly_stage = 4
        assembly_flag = False
        list_text_dict["assembly_stage"] = assembly_stage
        list_text_dict["description"] = "Please place the assembled part as shown."
        list_text_dict["ref_image"] = r"C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\ref_4.png"
    
    elif similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_2.jpg', image) >= ins_threshold:
        overlay = frame.copy()
        x, y, w, h =  525, 60, 200, 300  # Rectangle parameters
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1)
        p_x, p_y, p_w, p_h = 565, 565, 75, 60
        cv2.rectangle(overlay, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 200, 0), -1)
        new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        new_frame = cv2.arrowedLine(new_frame, (600, 565), (int(x + 0.5*w), y + h),
                                    (0,255,255), 3)

        assembly_stage = 3
        assembly_flag = False
        list_text_dict["assembly_stage"] = assembly_stage
        list_text_dict["description"] = "Place the Part 2 carefully in the bin."
        list_text_dict["ref_image"] = r"C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\ref_1.png"

    elif similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_3.jpg', image) >= ins_threshold:
        overlay = frame.copy()
        x, y, w, h =  800, 60, 200, 300  # Rectangle parameters
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1) 
        p_x, p_y, p_w, p_h = 569, 569, 45, 60
        cv2.rectangle(overlay, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 200, 0), -1)
        # print(Message)
        new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        new_frame = cv2.arrowedLine(new_frame, (600, 565), (int(x + 0.5*w), y + h), 
                                    (0,255,255), 3)

        assembly_stage = 2
        assembly_flag = False
        list_text_dict["assembly_stage"] = assembly_stage
        list_text_dict["description"] = " Place the Pi carefully in the bin."
        list_text_dict["ref_image"] = r"C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\ref_2.png"


    
    elif (disassembly_op1_similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_4.jpg', image) < ins_threshold_op_4 and
        similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_3.jpg', image) < ins_threshold and
        similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_2.jpg', image) < ins_threshold and 
        similarity(r'C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\inspection\\stage_1.jpg', image)):
        
        if assembly_flag == False and assembly_stage > 1:
            assembly_stage = max(0, assembly_stage)
            assembly_flag = True
            list_text_dict["assembly_stage"] = assembly_stage
            list_text_dict["description"] = " Check if the part is correctly assembled as shown or Move your hands away for inspection."
            list_text_dict["ref_image"] = list_text_dict["ref_image"]
        else:
            assembly_stage = max(0, assembly_stage)
            assembly_flag = True
            list_text_dict["assembly_stage"] = assembly_stage
            list_text_dict["description"] = "Check if the part is correctly assembled as shown or Move your hands away for inspection."
            list_text_dict["ref_image"] = list_text_dict["ref_image"]
        return new_frame, [list_text_dict["assembly_stage"], list_text_dict["description"], list_text_dict["ref_image"]], assembly_stage, assembly_flag, list_text_dict


    return new_frame, [list_text_dict["assembly_stage"], list_text_dict["description"], list_text_dict["ref_image"]], assembly_stage, assembly_flag, list_text_dict



if __name__ =="__main__":

    pass
