from os import close
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

img = cv2.imread("credimages/crop2.png")


# img  = cv2.resize(img,(1920,1080))
lap = img.copy()
laplacian = cv2.Laplacian(img,cv2.CV_64F)
cv2.imwrite("Laplacian.jpg",laplacian)

test  = cv2.imread("Laplacian.jpg")


# median = cv2.medianBlur(laplacian, 5)
test = cv2.bitwise_not(test)




canny = cv2.Canny(test,150,200,3)


anded = cv2.bitwise_and(img,img,mask=canny)
anded = cv2.bitwise_not(anded)
# kernel = np.ones((11,11))
# kernel1 = np.ones((3,3))
# opening = cv2.morphologyEx(test, cv2.MORPH_OPEN, kernel)
# opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel1)
# opening = cv2.medianBlur(opening,3)

# gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
# contours,_ = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


# for c in contours:
#     perimeter = cv2.arcLength(c,True) 
#     ROI = cv2.approxPolyDP(c,0.02*perimeter,True)

#     cv2.drawContours(c, contours, -1, (0, 0, 0), 5)

# closing = cv2.morphologyEx(test, cv2.MORPH_CLOSE, kernel)
# seed = (1004,1086)

# cv2.floodFill(test, None, seedPoint=seed, newVal=(255,255,255), loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
# cv2.circle(test, seed, 2, (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
    
cv2.imshow("Test",test)
cv2.imshow("Test1",laplacian)
cv2.imshow("Test11",anded)
# cv2.imshow("Test111",closing)

# cv2.imshow("Testope",canny)



text = pytesseract.image_to_string(test,lang='eng')
text = text.lower()
print(text)

d = pytesseract.image_to_data(test,output_type=Output.DICT)

n_boxes = len(d['text'])
for i in range(n_boxes):


    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    ocr_img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow("Testope1",img)

cv2.waitKey(0)