from os import close
import cv2
import numpy as np
from numpy.lib.type_check import imag
import pytesseract
from pytesseract import Output

img = cv2.imread("/Users/anishpawar/GID_9_2021/X_Ray_Anonimiser/Omega_Testing/laplacian++/credimages/amex_gold-min.png")


# img  = cv2.resize(img,(1920,1080))
lap = img.copy()
laplacian = cv2.Laplacian(img,cv2.CV_64F)
cv2.imwrite("Laplacian.jpg",laplacian)

test  = cv2.imread("Laplacian.jpg")

kernel = np.ones((3,3))

gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
opening = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
ret,thresh1 = cv2.threshold(opening,150,255,cv2.THRESH_BINARY)

canny = cv2.Canny(test,150,200,3)




kernel1 = np.ones((3,3))


anded = cv2.bitwise_and(test,opening,mask=thresh1)

contours,_ = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(test, contours, -1, (0, 0, 0), 9)
closing = cv2.morphologyEx(test, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Test",thresh1)
cv2.imwrite("Laplacian.jpg",thresh1)

text = pytesseract.image_to_string(thresh1,lang='eng')
text = text.lower()
print(text)

d = pytesseract.image_to_data(thresh1,output_type=Output.DICT)

n_boxes = len(d['text'])
for i in range(n_boxes):


    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    ocr_img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Testope1",img)
cv2.waitKey(0)