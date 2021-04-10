import cv2
import numpy as np
# from matplotlib import pyplot as plt

img = cv2.imread('good_images/visa_platinum-min.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()

# notted = cv2.bitwise_not(s)

ret,thresh1 = cv2.threshold(s,100,255,cv2.THRESH_BINARY)

#mega_and = cv2.bitwise_and(h, cv2.bitwise_and(s, v))

cv2.imshow("KK",h)
cv2.imshow("KK1",s)
cv2.imshow("KK1",thresh1)
cv2.imwrite("S.jpg",thresh1)


# anded = cv2.bitwise_and(s, v)
# cv2.imshow("KK2",v)
#cv2.imshow("KK_anded", mega_and)
# height = img.shape[0]
# width = img.shape[1]

# seed1= (int(width*0.30),int(height*0.30))
# seed2= (int(width*0.70),int(height*0.30))
# seed3= (int(width*0.70),int(height*0.70))
# seed4= (int(width*0.30),int(height*0.70))



# cv2.floodFill(s, None, seedPoint=seed1, newVal=(0,0,0), loDiff=(30,30,30,30), upDiff=(30,30,30,30))
# cv2.circle(s, seed1, 2, (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
# cv2.floodFill(s, None, seedPoint=seed2, newVal=(0,0,0), loDiff=(30,30,30,30), upDiff=((30,30,30,30)))
# cv2.circle(s, seed2, 2, (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
# cv2.floodFill(s, None, seedPoint=seed3, newVal=(0,0,0), loDiff=(30,30,30,30), upDiff=(30,30,30,30))
# cv2.circle(s, seed3, 2, (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
# cv2.floodFill(s, None, seedPoint=seed4, newVal=(0,0,0), loDiff=(30,30,30,30), upDiff=(30,30,30,30))
# cv2.circle(s, seed1, 4, (0, 0, 0), cv2.FILLED, cv2.LINE_AA)

# cv2.imshow("KKS",s)

cv2.waitKey(0)