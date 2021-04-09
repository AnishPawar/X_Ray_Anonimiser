import cv2
import numpy as np
import pytesseract
from pytesseract import Output

img = cv2.imread("credimages/ecoin-wirex.png")


img  = cv2.resize(img,(1920,1080))
lap = img.copy()
laplacian = cv2.Laplacian(img,cv2.CV_64F)
cv2.imwrite("Laplacian.jpg",laplacian)

test  = cv2.imread("Laplacian.jpg")

median = cv2.bilateralFilter(test,9,45,45)
canny = cv2.Canny(median,40,200,3)




seed = (546,325)

cv2.floodFill(img, None, seedPoint=seed, newVal=(255,255,255), loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
cv2.circle(img, seed, 2, (0, 0, 0), cv2.FILLED, cv2.LINE_AA)

# laplacian1 = cv2.Laplacian(median_not,cv2.CV_64F)

# ret, median_not = cv2.threshold(median, 20, 255, cv2.THRESH_BINARY_INV)

# median_not = cv2.cvtColor(median_not, cv2.COLOR_BGR2GRAY)
# contours, hierarchy = cv2.findContours(median_not, 
#     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# print(median_not.shape[0]*median_not.shape[1])
# image_area = (median_not.shape[0]*median_not.shape[1])
# To prevent from detecting the whole page as ROI
# for x in contours:
#     print(cv2.contourArea(x))
#     if cv2.contourArea(x) > 0.6*image_area:
#         contours.remove(x)

# cv2.drawContours(lap, contours, -1, (255, 0, 0), 5)

# img = cv2.bitwise_not(img)
cv2.imshow("Test",img)
cv2.imshow("Test1",laplacian)
cv2.imshow("Test11",median)

cv2.imshow("Testope",canny)



text = pytesseract.image_to_string(img,lang='eng')
text = text.lower()
print(text)

d = pytesseract.image_to_data(img,output_type=Output.DICT)

n_boxes = len(d['text'])
# for i in range(n_boxes):


#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     ocr_img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# cv2.imshow("Testope1",img)

cv2.waitKey(0)