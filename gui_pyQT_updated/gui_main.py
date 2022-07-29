from dis import disassemble
from operator import getitem
import sys
from xmlrpc.client import boolean
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QTimer

import time
from PyQt5 import QtGui
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.pyplot import text
from sklearn.feature_extraction import image
from PIL import Image, ImageTk
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
from threading import Thread
import tkinter

from gui_hand_tracking import detect_hands
from gui_inspect_new import Inspect, similarity, Disassemble_Inspect
import sqlite3


import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import pyrealsense2 as rs
import numpy as np
from gui_object_localisation import obj_detection




class VideoThread(QThread):
    
# VideoThread class passes the image and text decriptions to display in the application.

    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_indicator = pyqtSignal(list)

    

    def run(self):
        # Loading the trained TF SSD Mobnet model

        CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
        PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
        PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
        TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
        LABEL_MAP_NAME = 'label_map.pbtxt'

        paths = {
        'WORKSPACE_PATH': os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow','scripts'),
        'APIMODEL_PATH': os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow','models'),
        'ANNOTATION_PATH': os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','annotations'),
        'IMAGE_PATH': os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','images'),
        'MODEL_PATH': os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models'),
        'PRETRAINED_MODEL_PATH': os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
        'OUTPUT_PATH': os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
        'TFJS_PATH':os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
        'TFLITE_PATH':os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
        'PROTOC_PATH':os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow','protoc')
        }

        files = {
        'PIPELINE_CONFIG':os.path.join('C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
        }
        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])


        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()
        
        # Initiating camera feed and variables

        cap = cv2.VideoCapture(0)
        assembly_stage = 0
        assembly_flag = False
        list_texts = ["FAIL", "FAIL", "FAIL", "FAIL", "FAIL"]
        
        # Location of 3 Bin in the image. This is helpful for cropping the image and localise the parts.
        part_bbox_config = { 1: {"bbox" : [66,261, 336, 221]},
                    2: {"bbox" : [60,526, 331, 196]},
                    3: {"bbox" : [66, 794, 325, 202]}}

        y_1, x_1, h_1, w_1 = part_bbox_config[1]['bbox']
        y_2, x_2, h_2, w_2 = part_bbox_config[2]['bbox']
        y_3, x_3, h_3, w_3 = part_bbox_config[3]['bbox']
        
        # This list consists of all the information required for updating the display texts and images in the application.
        list_text_dict = {"assembly_stage":assembly_stage, 
                            "description":"Place the Part 2 in the Fixture as shown",
                            "ref_image":"FAIL"}
        
        

        while True:

            # getting the RGB image from the camera feed
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()

            color_image = np.asanyarray(color_frame.get_data())
            frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            image_np_1 = np.asanyarray(frame)[y_1:y_1 + h_1, x_1:x_1 + w_1,:]
            image_np_2 = np.asanyarray(frame)[y_2:y_2 + h_2, x_2:x_2 + w_2,:]
            image_np_3 = np.asanyarray(frame)[y_3:y_3 + h_3, x_3:x_3 + w_3,:]
            
            # localising the parts
            
            image_np_1  = obj_detection(image_np_1, detection_model, category_index)
            image_np_2 = obj_detection(image_np_2, detection_model, category_index)
            image_np_3 = obj_detection(image_np_3, detection_model, category_index)

            # updating the camera feed with the detected parts
            frame[y_1:y_1 + h_1, x_1:x_1 + w_1,:] = image_np_1
            frame[y_2:y_2 + h_2, x_2:x_2 + w_2,:] = image_np_2
            frame[y_3:y_3 + h_3, x_3:x_3 + w_3,:] = image_np_3

            print(self.welcome_screen.assemble_seq)
            
            # Detecting the hands
            frame, presence_hands = detect_hands(frame)
            print(".--------------------hands--------------", presence_hands)
            
            # Function for checking the assembly sequence
            if self.welcome_screen.assemble_seq:
            
                if presence_hands > 0:
                    frame, list_texts, assembly_stage, assembly_flag, list_text_dict = Inspect( frame = frame, image = frame[543:543+105, 549:549+106], list_text_dict = list_text_dict, assembly_stage = assembly_stage, assembly_flag = assembly_flag)
            # print("-------- Running Assemble--------", self.assemble_seq)
                else :
                    list_texts[1] = "Waiting for Worker"

            # Fucntion for checking the dissamble sequence

            else: 

                if presence_hands > 0:
                    frame, list_texts, assembly_stage, assembly_flag, list_text_dict = Disassemble_Inspect( frame = frame, image = frame[543:543+105, 549:549+106], list_text_dict = list_text_dict, assembly_stage = assembly_stage, assembly_flag = assembly_flag)
            # print("-------- Running Assemble--------", self.assemble_seq)
                else :
                    list_texts[1] = "Waiting for Worker"

            list_text_dict['assembly_stage'] = assembly_stage
            # else:
                # frame, list_texts, assembly_stage, assembly_flag = Disassemble_Inspect(frame = frame, image = frame[543:543+105, 549:549+106], list_text_dict = list_text_dict, assembly_stage = assembly_stage, assembly_flag = assembly_flag)
                # print("-------- Running Disasseble------", self.assemble_seq)
            print(list_texts)
            self.change_pixmap_signal.emit(frame)
            self.change_indicator.emit(list_texts)


            # print("Qt sender : ", QMainWindow.sender())


class WelcomeScreen(QMainWindow):
    """WelcomeScreem is a class. It loads the initial GUI screen using the file "gui_screen.ui". All the labels and pixel maps can be initiated and accessed via this class. It has 3 methods.
    1. update_indicators - This method is used to update the decription text and progress of assembly/disassembly stages.
    2. update_image - This method is used to update the camerafeed. 
    3. change_ref_image - This method is used to change the reference image according the assemble/disassembly stage.

    """
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        loadUi(r"C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\gui_pyQT_updated\gui_screen.ui",self)
        filename = r"C:\\Users\\Bakthavatchalam\\LF171_Werker_Assistent_System\\Scripts\\gui_pyQT\iwt_logo.png"
        
        self.IWT_logo_label.setPixmap(QPixmap(filename))

        self.assemble_seq = True
        self.assemble_pushButton.setCheckable(True)
        self.assemble_pushButton.setStyleSheet("color : white;font-size: 15pt;background-color: green;")
        self.disassemble_pushButton.clicked.connect(self.on_state_change_disassemble)
        self.assemble_pushButton.clicked.connect(self.on_state_change_assemble)


        print("assemble_seq from checkbox", self.assemble_seq)
        self.thread = VideoThread()
        self.thread.welcome_screen = self

        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_indicator.connect(self.update_indicators)

        self.thread.start()
        

        # timer = QTimer(self)
        # timer.timeout.connect(self.show_reference_image_2)
        # timer.start(500)

        self.disply_width = 600
        self.display_height = 400
        self.ref_img_width = 300
        self.ref_img_height = 200

    @pyqtSlot(list)
    def update_indicators(self, list_texts):
        """Updates the text labels and the appearnce of progress bar"""

        if list_texts[0] == 0:
            self.operation_1.setStyleSheet("color : black;font-size: 18pt;background-color: red;border: 3px solid black;border-radius: 40px;")
            self.operation_2.setStyleSheet("color : black;font-size: 18pt;background-color: white;border: 3px solid black;border-radius: 40px;")
            self.operation_3.setStyleSheet("color : black;font-size: 18pt;background-color: white;border: 3px solid black;border-radius: 40px;")
            self.operation_4.setStyleSheet("color : black;font-size: 18pt;background-color: white;border: 3px solid black;border-radius: 40px;")

            if list_texts[1] == "Waiting for Worker":
                self.operation_1.setStyleSheet("color : white;font-size: 18pt;background-color: orange;border: 3px solid black;border-radius: 40px;")

        elif list_texts[0] == 1 :

            self.reference_image_label.setPixmap(QtGui.QPixmap(list_texts[2]))
            self.operation_1.setStyleSheet("color : white;font-size: 18pt;background-color: red;border: 3px solid black;border-radius: 40px;")
            self.operation_description.setText(list_texts[1])
            self.operation_2.setStyleSheet("color : black;font-size: 18pt;background-color: white;border: 3px solid black;border-radius: 40px;")
            self.operation_3.setStyleSheet("color : black;font-size: 18pt;background-color: white;border: 3px solid black;border-radius: 40px;")
            self.operation_4.setStyleSheet("color : black;font-size: 18pt;background-color: white;border: 3px solid black;border-radius: 40px;")
            if list_texts[1] == "Waiting for Worker":
                self.operation_1.setStyleSheet("color : white;font-size: 18pt;background-color: orange;border: 3px solid black;border-radius: 40px;")

        elif list_texts[0] == 2:

            self.reference_image_label.setPixmap(QtGui.QPixmap(list_texts[2]))
            self.operation_1.setStyleSheet("color : white;font-size: 18pt;background-color: green;border: 3px solid black;border-radius: 40px;")
            self.operation_2.setStyleSheet("color : white;font-size: 18pt;background-color: red;border: 3px solid black;border-radius: 40px;")
            self.operation_description.setText(list_texts[1])

            self.operation_3.setStyleSheet("color : black;font-size: 18pt;background-color: white;border: 3px solid black;border-radius: 40px;")
            self.operation_4.setStyleSheet("color : black;font-size: 18pt;background-color: white;border: 3px solid black;border-radius: 40px;")

            if list_texts[1] == "Waiting for Worker":
                self.operation_2.setStyleSheet("color : white;font-size: 18pt;background-color: orange;border: 3px solid black;border-radius: 40px;")


            

        elif list_texts[0] == 3:

            self.reference_image_label.setPixmap(QtGui.QPixmap(list_texts[2]))
            self.operation_2.setStyleSheet("color : white;font-size: 18pt;background-color: green;border: 3px solid black;border-radius: 40px;")
            self.operation_3.setStyleSheet("color : white;font-size: 18pt;background-color: red;border: 3px solid black;border-radius: 40px;")
            self.operation_description.setText(list_texts[1])

            self.operation_1.setStyleSheet("color : white;font-size: 18pt;background-color: green;border: 3px solid black;border-radius: 40px;")
            self.operation_4.setStyleSheet("color : black;font-size: 18pt;background-color: white;border: 3px solid black;border-radius: 40px;")
            if list_texts[1] == "Waiting for Worker":
                self.operation_3.setStyleSheet("color : white;font-size: 18pt;background-color: orange;border: 3px solid black;border-radius: 40px;")
            

        elif list_texts[0] == 4:

            self.reference_image_label.setPixmap(QtGui.QPixmap(list_texts[2]))
            self.operation_3.setStyleSheet("color : white;font-size: 18pt;background-color: green;border: 3px solid black;border-radius: 40px;")
            self.operation_4.setStyleSheet("color : white;font-size: 18pt;background-color: red;border: 3px solid black;border-radius: 40px;")
            self.operation_description.setText(list_texts[1])

            self.operation_1.setStyleSheet("color : white;font-size: 18pt;background-color: green;border: 3px solid black;border-radius: 40px;")
            self.operation_2.setStyleSheet("color : white;font-size: 18pt;background-color: green;border: 3px solid black;border-radius: 40px;")
            if list_texts[1] == "Waiting for Worker":
                self.operation_4.setStyleSheet("color : white;font-size: 18pt;background-color: orange;border: 3px solid black;border-radius: 40px;")




    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.camera_feed.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def change_ref_image(self, file_name):
        """ Change the reference image in the QT application"""
        ref_image = cv2.imread(file_name)
        rgb_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.ref_img_width, self.ref_img_height, Qt.KeepAspectRatio)
        self.reference_image_label.setPixmap(QPixmap.fromImage(p))


    def on_state_change_assemble(self):
        """Change the appearance of disassemble button when assemble sequence is clicked"""
        self.disassemble_pushButton.setCheckable(False)
        self.assemble_seq = True
        self.disassemble_pushButton.setStyleSheet("color : black;font-size: 15pt;background-color: light grey;")
        self.assemble_pushButton.setStyleSheet("color : white;font-size: 15pt;background-color: green;")

    def on_state_change_disassemble(self):
        """Change the appearance of assemble button when disassemble sequence is clicked"""
        self.assemble_pushButton.setCheckable(False)
        self.assemble_seq = False 
        self.assemble_pushButton.setStyleSheet("color : black;font-size: 15pt;background-color: light grey;")
        self.disassemble_pushButton.setStyleSheet("color : white;font-size: 15pt;background-color: green;")




  



if __name__ == "__main__":




    ###############################
    assembly_stage = 1
    app = QApplication(sys.argv)
    welcome = WelcomeScreen()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(welcome)
    widget.setFixedHeight(900)
    widget.setFixedWidth(1550)
    pipeline = rs.pipeline()
    config = rs.config()
    canvas_w = 1280
    canvas_h = 800
    config.enable_stream(rs.stream.color, canvas_w, canvas_h, rs.format.bgr8, 30)
    cfg = pipeline.start(config)
    dev = cfg.get_device()
    photo=None
    widget.show()

    QApplication.processEvents()
    
    try:
        sys.exit(app.exec_())
    except:
        print("Exiting")
