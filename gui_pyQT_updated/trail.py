
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

before = cv2.imread(r'D:\LF171_Werker_Assistent_System\Scripts\inspection\\stage_3.jpg')[27:87,12:66,:]
after = cv2.imread(r'D:\LF171_Werker_Assistent_System\Scripts\inspection\\stage_4.jpg')[27:87,12:66,:]

# after = current_image 
# before = cv2.imread(ref_image_path)
after = cv2.GaussianBlur(after,(3,3),cv2.BORDER_DEFAULT)
before = cv2.GaussianBlur(before,(3,3),cv2.BORDER_DEFAULT)

before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

(similarity_score, diff) = structural_similarity(before_gray, after_gray, full=True)
diff = (diff * 255).astype("uint8")

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = contours[0] if len(contours) == 2 else contours[1]
# print("SIMILARITY SCORE : ", similarity_score)
print(similarity_score)

plt.imshow(diff)
plt.show()