import cv2
import numpy as np


import pytesseract
from pytesseract import Output

counter = 0
    
final_coor = []

# scanned = np.zeros((), np.uint8)

flag = 0 


def blur_function(img):
    blur_base = img 
    blur = cv2.GaussianBlur(blur_base,(25,25),0)
    cv2.imshow('image',blur)
    

def ocr_function(base,og):

    temp = base.copy()

    text = pytesseract.image_to_string(temp,lang='eng')
    
    d = pytesseract.image_to_data(base,output_type=Output.DICT)

    # Text Bounding Boxes
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 30:
            if d['text'][i] == "THAKKAR" or d['text'][i] == "VIRAJ":
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                ocr_img = cv2.rectangle(og, (x, y), (x + w, y + h), (0, 0, 0), -1)

            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            ocr_img = cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Processed',temp)
    return(text,og)

def mouse(event,x,y,z,w):
    
    global counter,final_coor 
    marked = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        counter = counter+1

        if counter == 1:
            pos1=(y,x)
            pos1_flip = (x,y)

            # final_coor.append(pos1_flip)
            final_coor.insert(0,pos1_flip)
            
            
            cv2.circle(marked,pos1_flip,60,(0,255,0),-1)
            cv2.imshow('image',marked)

        if counter == 2:
            pos2=(y,x)
            pos2_flip = (x,y)
            
            # final_coor.append(pos2_flip)
            final_coor.insert(1,pos2_flip)            
            
            cv2.circle(marked,pos2_flip,60,(0,255,0),-1)
            cv2.imshow('image',marked)

        if counter == 3:
            pos3=(y,x)
            pos3_flip = (x,y)
            
            # final_coor.append(pos3_flip)
            
            final_coor.insert(3,pos3_flip)

            cv2.circle(marked,pos3_flip,60,(0,255,0),-1)
            cv2.imshow('image',marked)

        if counter == 4:
            pos4=(y,x)      
            pos4_flip = (x,y)
            
            # final_coor.append(pos4_flip)

            final_coor.insert(2,pos4_flip)
            
            cv2.circle(marked,pos4_flip,60,(0,255,0),-1)
            print(final_coor)
            
            warp_function(final_coor,img)
            

            
def warp_function(coor,img):
    pts1 = np.float32(coor)
    pts2 = np.float32([(0, 0), (500, 0), (0, 600), (500, 600)])

    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    # global warped
    warped = cv2.warpPerspective(img, matrix, (500,600))            
    cv2.imshow("image",warped)
    # cv2.imwrite('Warped.jpg',warped)
    # scan(warped)
    
    global flag
    flag = 1




# def scan(crop):
#     global scanned
#     scanned = crop.copy()
    # scanned = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((11,11))

    # blurred = cv2.bilateralFilter(scanned,9,75,75)

    # scanned = cv2.adaptiveThreshold(scanned,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,11)
    # # closing = cv2.morphologyEx(scanned, cv2.MORPH_CLOSE, kernel)
    # # scanned = cv2.Canny(scanned,150,200)
    # ret2,th2 = cv2.threshold(scanned,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # scanned = cv2.medianBlur(scanned,11)

    # cv2.imshow('scanned',scanned)
    
def auto_crop(img):
    

    aspect_ratio = img.shape[1]/img.shape[0]

    height = int(1500/aspect_ratio)

    img = cv2.resize(img, (1500, height))

    # print(aspect_ratio)

    # pre processing

    blur = cv2.GaussianBlur(img,(5,5),0)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th4 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,3)

    closing = cv2.morphologyEx(th4,cv2.MORPH_CLOSE,np.ones((11,11)),iterations= 1)

    canny = cv2.Canny(closing,0,200,3)

    # Finding area of image
    img_area = img.shape[0]*img.shape[1]

    contours,_ = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # To prevent from detecting the whole page as ROI
    for x in contours:
        if cv2.contourArea(x) > 0.9*img_area:
            contours.remove(x)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    max=contours[max_index]

    # approximating polygon
    perimeter = cv2.arcLength(max,True) 
    ROI = cv2.approxPolyDP(max,0.02*perimeter,True)


    if len(ROI)==4:
        reshaped = ROI.reshape(4,2)

        summation = [(x+y) for x,y in reshaped]
        difference = [(x-y) for x,y in reshaped]

        br_index = np.argmax(summation)
        tl_index = np.argmin(summation)
        tr_index = np.argmax(difference)
        bl_index = np.argmin(difference)

        tl = reshaped[tl_index]
        tr = reshaped[tr_index]
        bl = reshaped[bl_index]
        br = reshaped[br_index]

        # Warping
        pts1 = np.float32([tl,tr,bl,br])
        pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])

        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        warped = cv2.warpPerspective(img, matrix, (500,600))            
        
        cv2.imshow("image",warped)
        print(len(ROI))
        return(warped)
    
    return(img)
    