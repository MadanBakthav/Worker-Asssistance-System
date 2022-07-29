import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp


def detect_hands(color_image):

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands


    pt_1 = np.array([(3, 87), (5, 364), (168, 365), (169, 86)], np.int32)
    pt_1 = pt_1.reshape((-1, 1, 2))
    pt_2 = np.array([(264, 86), (280, 370), (461, 372), (456, 80)], np.int32)
    pt_2 = pt_2.reshape((-1, 1, 2))
    pt_3 = np.array([(538, 67), (531, 371), (709, 375), (713, 73)], np.int32)
    pt_3 = pt_3.reshape((-1, 1, 2))
    pt_4 = np.array([(805, 63), (811, 362), (979, 369), (989, 74)], np.int32)
    pt_4 = pt_4.reshape((-1, 1, 2))
    pt_5 = np.array([(1093, 63), (1084, 344), (1271, 343), (1270, 66)], np.int32)
    pt_5 = pt_5.reshape((-1, 1, 2))


    with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.2,
    # static_image_mode=True,
    min_tracking_confidence=0.2) as hands:

        image = color_image.copy()
        image.flags.writeable = False
        image_height, image_width, channels  = color_image.shape
#         print(color_image.shape)
        
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


#         image = cv2.flip(image, 1)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         print(results)
        word = ""
        if results.multi_hand_landmarks:
#             print(results.multi_handedness[0].classification[0].label)
            for hand_landmarks in results.multi_hand_landmarks:
#                 print(
#                   f'Wrist lcocation: (',
#                   f'{hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width}, '
#                   f'{hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height})'
#                 )
                hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width  
                hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
                
                # if cv2.pointPolygonTest(pt_1, (hand_x,hand_y), False) != -1 : #bin_1.contains(point):
                #     word = "Bin 1"

                # elif cv2.pointPolygonTest(pt_2, (hand_x,hand_y), False) != -1 : #bin_2.contains(point):
                #     word = "Bin 2"

                # elif cv2.pointPolygonTest(pt_3, (hand_x,hand_y), False) != -1 : #bin_3.contains(point):
                #     word = "Bin 3"

                # elif cv2.pointPolygonTest(pt_4, (hand_x,hand_y), False) != -1 : #bin_4.contains(point):
                #     word = "Bin 4"

                # elif cv2.pointPolygonTest(pt_5, (hand_x,hand_y), False) != -1 : #bin_5.contains(point):
                #     word = "Bin 5"
                    

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
#         cv2.imwrite("realsense_image.jpg",image)
        # cv2.putText(img=image, text=word, org=(int(0.8 * image_width),int(0.075 * image_height)), 
        #             fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)
    
        # image = cv2.polylines(image, [pt_1], True, (255, 0, 0), 2)

        # image = cv2.polylines(image, [pt_2], True, (255, 0, 0), 2)

        # image = cv2.polylines(image, [pt_3], True, (255, 0, 0), 2)

        # image = cv2.polylines(image, [pt_4], True, (255, 0, 0), 2)

        # image = cv2.polylines(image, [pt_5], True, (255, 0, 0), 2)

        return image


# pipeline.stop()



if __name__=="__main__":
    pass