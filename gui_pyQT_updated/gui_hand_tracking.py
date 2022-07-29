import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp


def detect_hands(color_image):
# initiating the classes for drawing and visualising the hands in the iamge
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

# location of Bins with respect to the camera. this can be helpful to check if the hands are inside the bin or not.
#     pt_1 = np.array([(3, 87), (5, 364), (168, 365), (169, 86)], np.int32)
#     pt_1 = pt_1.reshape((-1, 1, 2))
#     pt_2 = np.array([(264, 86), (280, 370), (461, 372), (456, 80)], np.int32)
#     pt_2 = pt_2.reshape((-1, 1, 2))
#     pt_3 = np.array([(538, 67), (531, 371), (709, 375), (713, 73)], np.int32)
#     pt_3 = pt_3.reshape((-1, 1, 2))
#     pt_4 = np.array([(805, 63), (811, 362), (979, 369), (989, 74)], np.int32)
#     pt_4 = pt_4.reshape((-1, 1, 2))
#     pt_5 = np.array([(1093, 63), (1084, 344), (1271, 343), (1270, 66)], np.int32)
#     pt_5 = pt_5.reshape((-1, 1, 2))

# Detecting the hands
    with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2) as hands:

        image = color_image.copy()
        image.flags.writeable = False
        image_height, image_width, channels  = color_image.shape
        no_of_hands = 0
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        word = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width  
                hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        if type(results.multi_hand_landmarks) == list:
                no_of_hands = len(results.multi_hand_landmarks)

        return image, no_of_hands




if __name__=="__main__":
    pass