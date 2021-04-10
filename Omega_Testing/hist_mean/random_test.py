from os import close
import cv2
import numpy as np
from numpy.lib.type_check import imag
import pytesseract
from pytesseract import Output

img = cv2.imread("S.jpg")

lap = img.copy()
laplacian = cv2.Laplacian(img,cv2.CV_64F)
cv2.imwrite("Laplacian.jpg",laplacian)

test  = cv2.imread("Laplacian.jpg")

gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

canny = cv2.Canny(test,150,200,3)

anded = cv2.bitwise_and(test,test,mask=thresh1)

kernel = np.ones((7,7))
kernel1 = np.ones((3,3))
opening = cv2.morphologyEx(test, cv2.MORPH_OPEN, kernel)

contours,_ = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(test, contours, -1, (0, 0, 0), 9)
closing = cv2.morphologyEx(test, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Test",img)

cv2.imwrite("Laplacian.jpg",img)

text = pytesseract.image_to_string(img,lang='eng')
text = text.lower()
print(text)

d = pytesseract.image_to_data(img,output_type=Output.DICT)

n_boxes = len(d['text'])
for i in range(n_boxes):

    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    ocr_img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Testope1",img)

cv2.waitKey(0)