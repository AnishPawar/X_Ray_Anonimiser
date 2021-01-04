import cv2
import numpy as np

img = cv2.imread('trace-expand.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray,(10,10))

img_area = img.shape[0]*img.shape[1]

kernel = np.ones((50,50))
eroded = cv2.erode(blur,kernel=kernel)

x,thresh = cv2.threshold(eroded,100,255,cv2.THRESH_BINARY)

contours,x = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
for x in contours:
    if cv2.contourArea(x) > 0.9*img_area:
        contours.remove(x)

counter = 0
for contour in contours:
    counter+=1
    x,y,w,h = cv2.boundingRect(contour)
    cropped = gray[y:(y+h),x:(x+w)]
    cv2.imwrite("Image_{}.jpg".format(counter),cropped)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)



cv2.imshow("IMage",img)
cv2.waitKey(0)