import cv2
import numpy as np
import os
import math

images = os.listdir("/Users/anishpawar/Git/X_Ray_Data_Detector/Omega_Testing/X-Ray-Images")

threshold = 90

for image in images:

    img = cv2.imread("X-Ray-Images/{}".format(image))
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    var = cv2.Laplacian(img, cv2.CV_64F).var()
    print(var)

    cv2.imshow("Image1",img)
    cv2.waitKey(0)


# sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#     og_img = cv2.filter2D(og_img, -1, sharpen_kernel)

#     kernel = np.ones((5,5),np.uint8)
#     og_img = cv2.morphologyEx(og_img, cv2.MORPH_OPEN, kernel)