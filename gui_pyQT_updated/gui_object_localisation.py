import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import threading
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def obj_detection (image, detection_model, category_index):
    """ This function takes the image, trained model and the labels for the detected objects
    as input.

    Args:
        image (np.array): Here, this image is the section of bin sliced from the camerafeed
        detection_model (TF SSD Model): it is trained standard tensorflow SSD model
        category_index (dict): Part labels for detected objects

    Returns:
        image (np.array): This image contains the detected objects in the input image.
    """

    image_np = np.array(image)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    len([x for x in detections['detection_scores'] if x > 0.8])

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    no_of_objects = len([x for x in detections['detection_scores'] if x > 0.8])
    cv2.putText(img=image_np_with_detections, text= str(no_of_objects), org=(100, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

    # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    # plt.show()
    return image_np_with_detections



if __name__ =="__main__":

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

    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()


    part_bbox_config = { 1: {"bbox" : [66,261, 336, 221]},
                        2: {"bbox" : [60,526, 331, 196]},
                        3: {"bbox" : [66, 794, 325, 202]}}



    y_1, x_1, h_1, w_1 = part_bbox_config[1]['bbox']
    # image_np_1 = np.asanyarray(image)[y_1:y_1 + h_1, x_1:x_1 + w_1,:]
    y_2, x_2, h_2, w_2 = part_bbox_config[2]['bbox']
    # image_np_2 = np.asanyarray(image)[y_2:y_2 + h_2, x_2:x_2 + w_2,:]
    y_3, x_3, h_3, w_3 = part_bbox_config[3]['bbox']
    # image_np_3 = np.asanyarray(image)[y_3:y_3 + h_3, x_3:x_3 + w_3,:]
    # image = cv2.flip(image, 1)
    # image_np_1 = obj_detection(image_np_1)
    # image_np_2 = obj_detection(image_np_2)
    # image_np_3 = obj_detection(image_np_3)

    # y, x,  h, w = part_bbox_config[1]['bbox']
    # image[y:y + h, x:x + w,:] = image_np_1
    # y, x,  h, w = part_bbox_config[2]['bbox']
    # image[y:y + h, x:x + w,:] = image_np_2
    # y, x,  h, w = part_bbox_config[3]['bbox']
    # image[y:y + h, x:x + w,:] = image_np_3
