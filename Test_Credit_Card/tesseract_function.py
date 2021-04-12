import pytesseract
import cv2
import numpy as np

results = []
def tesseacr_func(x,y,w,h,aspect_ratio,og_image,thresh):
    startX = int(x*aspect_ratio)
    startY =int(y*aspect_ratio) 
    endX = int((x+w)*aspect_ratio)
    endY = int((y+h)*aspect_ratio)

    kernel = np.ones((7,7))
    org = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    anded = cv2.bitwise_and(image,image,mask=opening)

    r = org[startY:endY, startX:endX]
    r = cv2.bitwise_not(r)

    # rk = org[startY:endY, startX:endX]
    # cv2.imshow("k",rk)
    # cv2.waitKey(0)

    configuration = ("-l eng --oem 1 --psm 8")
    text = pytesseract.image_to_string(r, config=configuration)
    print(text)
    results.append(((startX, startY, endX, endY), text))

    return results