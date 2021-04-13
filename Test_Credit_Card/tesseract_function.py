import pytesseract
import cv2
import numpy as np
import re

# results = []
def tesseacr_func(x,y,w,h,aspect_ratio,og_image):
    startX = int(x*aspect_ratio)
    startY =int(y*aspect_ratio) 
    endX = int((x+w)*aspect_ratio)
    endY = int((y+h)*aspect_ratio)
    r = og_image[startY:endY, startX:endX]
    r = cv2.bitwise_not(r)


    # cv2.imshow("KK",r)
    # cv2.waitKey(0)

    configuration = ("-l eng --oem 1 --psm 8")
    text = pytesseract.image_to_string(r, config=configuration)
    
    text = "".join(re.findall("[a-zA-Z]+", text))
    print(text)

    if not text.islower():

        results = (startX, startY, endX, endY, text)
        return results
    return