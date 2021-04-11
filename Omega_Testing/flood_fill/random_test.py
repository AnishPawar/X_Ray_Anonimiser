from os import close
import cv2
import numpy as np
from numpy.lib.type_check import imag
import pytesseract
from pytesseract import Output

img = cv2.imread("/Users/anishpawar/GID_9_2021/X_Ray_Anonimiser/Omega_Testing/flood_fill/credimages/visa.png")


# img  = cv2.resize(img,(1920,1080))
lap = img.copy()
laplacian = cv2.Laplacian(img,cv2.CV_64F)
cv2.imwrite("Laplacian.jpg",laplacian)

test  = cv2.imread("Laplacian.jpg")



# median = cv2.medianBlur(laplacian, 5)


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

# seed = (1004,1086)


# Test 55
# seed1 = (157,1017)
# seed2 = (172,119)
# seed3 = (1932,132)
# seed4 = (1851,1105)

# Test33
# seed1 = (158,1044)
# seed2 = (1610,1025)
# seed3 = (1804,316)
# seed4 = (132,338)

height = img.shape[0]
width = img.shape[1]

seed1= (int(width*0.30),int(height*0.30))
seed2= (int(width*0.70),int(height*0.30))
seed3= (int(width*0.70),int(height*0.70))
seed4= (int(width*0.30),int(height*0.70))



cv2.floodFill(img, None, seedPoint=seed1, newVal=(0,0,0), loDiff=(30,30,30,30), upDiff=(30,30,30,30))
cv2.circle(img, seed1, 2, (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
cv2.floodFill(img, None, seedPoint=seed2, newVal=(0,0,0), loDiff=(30,30,30,30), upDiff=((30,30,30,30)))
cv2.circle(img, seed2, 2, (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
cv2.floodFill(img, None, seedPoint=seed3, newVal=(0,0,0), loDiff=(30,30,30,30), upDiff=(30,30,30,30))
cv2.circle(img, seed3, 2, (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
cv2.floodFill(img, None, seedPoint=seed4, newVal=(0,0,0), loDiff=(30,30,30,30), upDiff=(30,30,30,30))
cv2.circle(img, seed1, 4, (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
    

img = cv2.bitwise_not(img)
cv2.imshow("Test",img)
# img = cv2.bitwise_not(img)
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