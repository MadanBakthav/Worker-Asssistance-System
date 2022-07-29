from matplotlib.contour import ContourSet
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import threading
import matplotlib.pylab as plt
from skimage.metrics import structural_similarity



def similarity(assembly_stage, current_image):
    ref_image_no = assembly_stage
    after = current_image
    before = cv2.imread(r'D:\LF171_Werker_Assistent_System\Scripts\inspection\\stage_'+ str(ref_image_no) +'.jpg')
    after = cv2.GaussianBlur(after,(3,3),cv2.BORDER_DEFAULT)
    before = cv2.GaussianBlur(before,(3,3),cv2.BORDER_DEFAULT)

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    (similarity_score, diff) = structural_similarity(before_gray, after_gray, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]
    return similarity_score, contours



def Inspect(assembly_stage, image, frame, list_texts = ["NOT_OK", "NOT_OK", "NOT_OK", "NOT_OK", "NOT_OK"]):

    alpha = 0.5
    list_texts = list_texts
    ins_threshold = 0.92
    Message = "NOT_OK"
    new_frame = frame
    if assembly_stage == 1 :
        similarity_score, contours= similarity(assembly_stage, image)
        last_cycle_score, contours= similarity(5, image)
        if similarity_score > ins_threshold:
            list_texts = ["OK", "NOT_OK", "NOT_OK", "NOT_OK", "NOT_OK"]
            overlay = frame.copy()
            x, y, w, h =  525, 60, 200, 300  # Rectangle parameters
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1)
            p_x, p_y, p_w, p_h = 565, 565, 75, 60
            cv2.rectangle(overlay, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 200, 0), -1)
            # print(Message)
            new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            new_frame = cv2.arrowedLine(new_frame, (int(x + 0.5*w), y + h), (600, 565),
                                     (0,255,255), 3)
            # x, y, w, h =  525, 60, 200, 300  # Rectangle parameters
            # cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1) 
            # # print(Message)
            # new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            # assembly_stage += 1
        elif last_cycle_score > ins_threshold:

            list_texts = ["OK", "OK", "OK", "OK", "OK"]
            # print(Message_nxt)
            assembly_stage = 1
        else:
            next_step_score, contours= similarity(2, image)
            if next_step_score > ins_threshold:
                list_texts = ["OK", "OK", "NOT_OK", "NOT_OK", "NOT_OK"]
                assembly_stage = 2
                overlay = frame.copy()
                x, y, w, h =  525, 60, 200, 300  # Rectangle parameters
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1)

                p_x, p_y, p_w, p_h = 569, 569, 55, 60
                cv2.rectangle(overlay, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 200, 0), -1)
                # print(Message)
                new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                new_frame = cv2.arrowedLine(new_frame, (int(x + 0.5*w), y + h), (600, 565),
                                     (0,255,255), 3)
            else:
                Message = "NOT_OK"
                # print(Message)
    elif assembly_stage == 2 :
        similarity_score, contours= similarity(assembly_stage, image)
        if similarity_score > ins_threshold:
            list_texts = ["OK", "OK", "NOT_OK", "NOT_OK", "NOT_OK"]
            # print(Message)
            # assembly_stage += 1

            overlay = frame.copy()
            x, y, w, h =  800, 60, 200, 300  # Rectangle parameters
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1) 
            p_x, p_y, p_w, p_h = 569, 569, 45, 60
            cv2.rectangle(overlay, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 200, 0), -1)
            # print(Message)
            new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            new_frame = cv2.arrowedLine(new_frame, (int(x + 0.5*w), y + h), (600, 565),
                                     (0,255,255), 3)
        else:
            # Message = "NOT OK"
            next_step_score, contours= similarity(3, image)
            if next_step_score > ins_threshold:
                # Message_nxt = "----------------------next_OK"
                # print(Message_nxt)
                list_texts = ["OK", "OK", "OK", "NOT_OK", "NOT_OK"]
                assembly_stage = 3
            else:
                Message = "NOT_OK"
                # print(Message)
                # assembly_stage += 1

    elif assembly_stage == 3 :
        similarity_score, contours= similarity(assembly_stage, image)
        if similarity_score > ins_threshold:
            Message = "OK"
            overlay = frame.copy()
            list_texts = ["OK", "OK", "OK", "NOT_OK", "NOT_OK"]
            x, y, w, h =  260, 60, 200, 300  # Rectangle parameters
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1) 
            p_x, p_y, p_w, p_h = 569, 569, 45, 60
            cv2.rectangle(overlay, (p_x, p_y), (p_x+p_w, p_y+p_h), (0, 200, 0), -1)
            # print(Message)
            new_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            new_frame = cv2.arrowedLine(new_frame, (int(x + 0.5*w), y + h), (600, 565),
                                     (0,255,255), 3)
            # print(Message)
            # assembly_stage += 1
        else:
            # Message = "NOT OK"
            next_step_score, contours= similarity(4, image)
            if next_step_score > ins_threshold:
                # Message_nxt = "----------------------next_OK"
                # print(Message_nxt)
                assembly_stage = 4
                list_texts = ["OK", "OK", "OK", "OK", "NOT_OK"]
            else: 
                Message = "NOT_OK"
                # print(Message)
    elif assembly_stage == 4 :
        similarity_score, contours= similarity(assembly_stage, image)
        if similarity_score > ins_threshold:
            list_texts = ["OK", "OK", "OK", "OK", "NOT_OK"]
            # print(Message)
            # assembly_stage = 1
        else:
            next_step_score, contours= similarity(1, image)
            if next_step_score > ins_threshold:
                # Message_nxt = "next_Cy"
                # print(Message_nxt)
                assembly_stage = 1
                list_texts = ["OK", "OK", "OK", "OK", "OK"]
            else: 
                Message = "NOT_OK"
                # print(Message)

    return new_frame, list_texts, assembly_stage, contours


if __name__ =="__main__":

    pass
