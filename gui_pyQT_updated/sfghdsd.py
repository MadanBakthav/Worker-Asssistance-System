from operator import getitem
import sys
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
from gui_inspect import Inspect, similarity
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
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_indicator = pyqtSignal(list)

    def run(self):
        # capture from web cam

        CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
        PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
        PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
        TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
        LABEL_MAP_NAME = 'label_map.pbtxt'

        paths = {
        'WORKSPACE_PATH': os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow','scripts'),
        'APIMODEL_PATH': os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow','models'),
        'ANNOTATION_PATH': os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','annotations'),
        'IMAGE_PATH': os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','images'),
        'MODEL_PATH': os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models'),
        'PRETRAINED_MODEL_PATH': os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
        'OUTPUT_PATH': os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
        'TFJS_PATH':os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
        'TFLITE_PATH':os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
        'PROTOC_PATH':os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow','protoc')
        }

        files = {
        'PIPELINE_CONFIG':os.path.join('D:\\LF171_Werker_Assistent_System\\Scripts\\yolo\\Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
        }
        # print(paths)
        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

        # config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

        cap = cv2.VideoCapture(0)
        assembly_stage = 1
        list_texts = ["NOT_OK", "NOT_OK", "NOT_OK", "NOT_OK", "NOT_OK"]
        part_bbox_config = { 1: {"bbox" : [66,261, 336, 221]},
                    2: {"bbox" : [60,526, 331, 196]},
                    3: {"bbox" : [66, 794, 325, 202]}}

        y_1, x_1, h_1, w_1 = part_bbox_config[1]['bbox']
        y_2, x_2, h_2, w_2 = part_bbox_config[2]['bbox']
        y_3, x_3, h_3, w_3 = part_bbox_config[3]['bbox']

        while True:

            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()

            color_image = np.asanyarray(color_frame.get_data())
            frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            image_np_1 = np.asanyarray(frame)[y_1:y_1 + h_1, x_1:x_1 + w_1,:]
            image_np_2 = np.asanyarray(frame)[y_2:y_2 + h_2, x_2:x_2 + w_2,:]
            image_np_3 = np.asanyarray(frame)[y_3:y_3 + h_3, x_3:x_3 + w_3,:]
            
            image_np_1 = obj_detection(image_np_1, detection_model, category_index)
            image_np_2 = obj_detection(image_np_2, detection_model, category_index)
            image_np_3 = obj_detection(image_np_3, detection_model, category_index)

            frame[y_1:y_1 + h_1, x_1:x_1 + w_1,:] = image_np_1
            frame[y_2:y_2 + h_2, x_2:x_2 + w_2,:] = image_np_2
            frame[y_3:y_3 + h_3, x_3:x_3 + w_3,:] = image_np_3


            frame = detect_hands(frame)
            frame, list_texts, assembly_stage, contours = Inspect(assembly_stage = assembly_stage, list_texts = list_texts, frame = frame, image = frame[543:543+105, 549:549+106])
            print(list_texts)
            self.change_pixmap_signal.emit(frame)
            self.change_indicator.emit(list_texts)



class WelcomeScreen(QMainWindow):
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        loadUi(r"D:\LF171_Werker_Assistent_System\Scripts\gui_pyQT\gui_screen.ui",self)
        filename = r"D:\LF171_Werker_Assistent_System\Scripts\gui_pyQT\iwt_logo.png"
        self.IWT_logo_label.setPixmap(QPixmap(filename))
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_indicator.connect(self.update_indicators)
        # pindicators
        # indicators.signal
        # start the thread
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
        if list_texts == ['OK', 'OK', 'NOT_OK', 'NOT_OK', 'NOT_OK']:
            self.indication_op1.setText(list_texts[1])
            self.indication_op1.setStyleSheet("color : green;font-size: 18pt")
            # self.change_ref_image(file_name=r"D:\LF171_Werker_Assistent_System\Scripts\inspection\stage_3.jpg")
            # self.change_ref_image(file_name=r"D:\LF171_Werker_Assistent_System\Scripts\inspection\\")
            self.reference_image_label.setPixmap(QtGui.QPixmap(r"D:\LF171_Werker_Assistent_System\Scripts\inspection\ref_3.png"))
            self.header_op1_label.setStyleSheet("color : black;font-size: 18pt;background-color: lightgreen")



            # self.show_reference_image_2()
            # QApplication.processEvents()
        if list_texts == ['OK', 'NOT_OK', 'NOT_OK', 'NOT_OK', 'NOT_OK']:
            self.indication_op1.setText(list_texts[1])
            self.indication_op2.setText("")
            self.indication_op3.setText("")
            self.indication_op4.setText("")
            self.indication_op1.setStyleSheet("color : red;font-size: 18pt")
            # self.change_ref_image(file_name=r"D:\LF171_Werker_Assistent_System\Scripts\inspection\stage_2.jpg")
            # self.change_ref_image(file_name=r"D:\LF171_Werker_Assistent_System\Scripts\inspection\\")
            # self.show_reference_image_1()
            # QApplication.processEvents()
            self.reference_image_label.setPixmap(QtGui.QPixmap(r"D:\LF171_Werker_Assistent_System\Scripts\inspection\ref_2.png"))
            self.header_op1_label.setStyleSheet("color : black;font-size: 18pt;background-color: yellow")
            self.header_op2_label.setStyleSheet("color : black;font-size: 18pt;background-color: white")
            self.header_op3_label.setStyleSheet("color : black;font-size: 18pt;background-color: white")
            self.header_op4_label.setStyleSheet("color : black;font-size: 18pt;background-color: white")


        if list_texts == ['OK', 'OK', 'OK', 'NOT_OK', 'NOT_OK']:
            self.indication_op2.setText(list_texts[2])
            self.indication_op2.setStyleSheet("color : green;font-size: 18pt")
            # self.change_ref_image(file_name=r"D:\LF171_Werker_Assistent_System\Scripts\inspection\stage_4.jpg")
            # self.show_reference_image_3()
            # QApplication.processEvents()
            self.reference_image_label.setPixmap(QtGui.QPixmap(r"D:\LF171_Werker_Assistent_System\Scripts\inspection\ref_4.png"))
            self.header_op2_label.setStyleSheet("color : black;font-size: 18pt;background-color: lightgreen")

        if list_texts == ['OK', 'OK', 'NOT_OK', 'NOT_OK', 'NOT_OK']:
            self.indication_op2.setText(list_texts[2])
            self.indication_op2.setStyleSheet("color : red;font-size: 18pt")
            # self.change_ref_image(file_name=r"D:\LF171_Werker_Assistent_System\Scripts\inspection\stage_3.jpg")
            # self.show_reference_image_2()
            # QApplication.processEvents()
            self.reference_image_label.setPixmap(QtGui.QPixmap(r"D:\LF171_Werker_Assistent_System\Scripts\inspection\ref_3.png"))
            self.header_op2_label.setStyleSheet("color : black;font-size: 18pt;background-color: yellow")


        if list_texts == ['OK', 'OK', 'OK', 'OK', 'NOT_OK']:
            self.indication_op3.setText(list_texts[3])
            self.indication_op3.setStyleSheet("color : green;font-size: 18pt")
            # self.change_ref_image(file_name=r"D:\LF171_Werker_Assistent_System\Scripts\inspection\stage_4.jpg")
            # self.show_reference_image_4()
            # QApplication.processEvents()
            self.reference_image_label.setPixmap(QtGui.QPixmap(r"D:\LF171_Werker_Assistent_System\Scripts\inspection\ref_4.png"))
            self.header_op3_label.setStyleSheet("color : black;font-size: 18pt;background-color: lightgreen")


        if list_texts == ['OK', 'OK', 'OK', 'NOT_OK', 'NOT_OK']:
            self.indication_op3.setText(list_texts[3])
            self.indication_op3.setStyleSheet("color : red;font-size: 18pt")
            # self.change_ref_image(file_name=r"D:\LF171_Werker_Assistent_System\Scripts\inspection\stage_5.jpg")
            # self.show_reference_image_3()
            # QApplication.processEvents()
            self.reference_image_label.setPixmap(QtGui.QPixmap(r"D:\LF171_Werker_Assistent_System\Scripts\inspection\ref_5.png"))
            self.header_op3_label.setStyleSheet("color : black;font-size: 18pt;background-color: yellow")

        if list_texts == ['OK', 'OK', 'OK', 'OK', 'OK']:
            self.indication_op4.setText(list_texts[4])
            self.indication_op4.setStyleSheet("color : green;font-size: 18pt")
            # self.change_ref_image(file_name=r"D:\LF171_Werker_Assistent_System\Scripts\inspection\stage_1.jpg")
            # self.show_reference_image_5()
            # QApplication.processEvents()
            self.reference_image_label.setPixmap(QtGui.QPixmap(r"D:\LF171_Werker_Assistent_System\Scripts\inspection\ref_1.png"))
            self.header_op4_label.setStyleSheet("color : black;font-size: 18pt;background-color: lightgreen")

        if list_texts == ['OK', 'OK', 'OK', 'OK', 'NOT_OK']:
            self.indication_op4.setText(list_texts[4])
            self.indication_op4.setStyleSheet("color : red;font-size: 18pt")
            # self.change_ref_image(file_name=r"D:\LF171_Werker_Assistent_System\Scripts\inspection\stage_1.jpg")
            self.reference_image_label.setPixmap(QtGui.QPixmap(r"D:\LF171_Werker_Assistent_System\Scripts\inspection\ref_1.png"))
            # self.show_reference_image_5()
            # QApplication.processEvents()
            self.header_op4_label.setStyleSheet("color : black;font-size: 18pt;background-color: yellow")



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
        ref_image = cv2.imread(file_name)
        rgb_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.ref_img_width, self.ref_img_height, Qt.KeepAspectRatio)
        self.reference_image_label.setPixmap(QPixmap.fromImage(p))




# main
if __name__ == "__main__":
    # main

    #####
    ##
    #####
    





    ###############################
    assembly_stage = 1
    app = QApplication(sys.argv)
    welcome = WelcomeScreen()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(welcome)
    widget.setFixedHeight(800)
    widget.setFixedWidth(1500)
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
