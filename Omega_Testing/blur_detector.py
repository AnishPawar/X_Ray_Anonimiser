import cv2
import numpy as np
import os
import math

images = os.listdir("/Users/anishpawar/Git/X_Ray_Data_Detector/Alpha_Testing/X-Ray-Images")

threshold = 90

for image in images:

    img = cv2.imread("X-Ray-Images/{}".format(image))
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    var = cv2.Laplacian(img, cv2.CV_64F).var()
    print(var)

    if var<=threshold:

        k_size = int((threshold - var))**2

        img_area = img.shape[0]*img.shape[1]

        if k_size**2 >= int(img_area/8):
            k_size = int(math.sqrt(img_area/3))

        if k_size%2 ==0:
            k_size+=1


        blur = cv2.GaussianBlur(img,(k_size,k_size),3)

        sharpened = cv2.addWeighted(img,1.5,blur,-0.5,0)

        var = cv2.Laplacian(sharpened, cv2.CV_64F).var()
        print(var)

        cv2.imshow("Image",sharpened)

    cv2.imshow("Image1",img)
    cv2.waitKey(0)