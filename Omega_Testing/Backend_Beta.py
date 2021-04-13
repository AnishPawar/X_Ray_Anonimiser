import cv2
import numpy as np


import pytesseract
from pytesseract import Output

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from fuzzywuzzy import fuzz as fz
from fuzzywuzzy import process as pr
import csv
import re

counter = 0
    
final_coor = []
temp_list = []
new_text = []
image = []

flag = 0 

def image_parser(img):
    global image
    image = img

def preprocess(image):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    return gray,thresh


def tesseacr_func(x,y,w,h,aspect_ratio,og_image):
    startX = int(x*aspect_ratio)
    startY =int(y*aspect_ratio) 
    endX = int((x+w)*aspect_ratio)
    endY = int((y+h)*aspect_ratio)
    r = og_image[startY:endY, startX:endX]
    r = cv2.bitwise_not(r)

    configuration = ("-l eng --oem 1 --psm 8")
    text = pytesseract.image_to_string(r, config=configuration)
    print(text)


        d = pytesseract.image_to_data(base,output_type=Output.DICT)

    processed_text = NLP(text)
    print(processed_text)

    for name in processed_text:
        fuzzy_matching(name)
    

    if not new_text:
        new_text = temp_list
        
    # print(new_text)    

    n_boxes = len(d['text'])
    for i in range(n_boxes):

        for c in new_text:
            comp = "".join(re.findall("[a-zA-Z0-9]+", d['text'][i].lower()))
            # c1 = ''.join(x for x in c if not x.isdigit())
            # print("HELLO -->", comp, '\n' , c, '\n')

            if  comp == c.lower():

                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                ocr_img = cv2.rectangle(og, (x, y), (x + w, y + h), (0, 0, 0), -1)

    results = (startX, startY, endX, endY, text)
    return results


def ocr_function(base,og):
    s = '1234567890'

    global new_text,temp_list
    temp = base.copy()

    text = pytesseract.image_to_string(temp,lang='eng')
    text = text.lower()
    print(text)

    d = pytesseract.image_to_data(base,output_type=Output.DICT)

    processed_text = NLP(text)
    print(processed_text)

    for name in processed_text:
        fuzzy_matching(name)
    

    if not new_text:
        new_text = temp_list
        
    # print(new_text)    

    n_boxes = len(d['text'])
    for i in range(n_boxes):

        for c in new_text:
            comp = "".join(re.findall("[a-zA-Z0-9]+", d['text'][i].lower()))
            # c1 = ''.join(x for x in c if not x.isdigit())
            # print("HELLO -->", comp, '\n' , c, '\n')

            if  comp == c.lower():

                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                ocr_img = cv2.rectangle(og, (x, y), (x + w, y + h), (0, 0, 0), -1)


    # cv2.imshow('Processed',temp)
    new_text = []
    return(text,og)

def NLP(text):
    global new_text,temp_list
    # Text Pre-Processing
    text = text.split(' ')

    new = []

    for i in text:
        i = i.strip()
        
        a = i.replace('\n','')
        a2 = "".join(re.findall("[a-zA-Z0-9]+", a))

        if len(i) > 2:
            new.append(a2.capitalize())
    processed = " ".join(new)

    ne_tree = pos_tag(word_tokenize(processed))
    print(ne_tree)
    names = []

    # Checking P-Noun Followed by P-Noun
    for c in range(len(ne_tree)):

        if ((ne_tree[c-1][1] == "NNP") and (ne_tree[c][1] == "NNP")) or ((ne_tree[c-1][1] == "NNP") and (ne_tree[c][1] == "NN")) or ((ne_tree[c-1][1] == "NN") and (ne_tree[c][1] == "NNP") or ne_tree[c][1] == "CD"):
            
            names.append(ne_tree[c-1][0])
            names.append(ne_tree[c][0])
    
    # If list is empty, return the original text
    if not names :
        for c in ne_tree:
            temp = "".join(re.findall("[a-zA-Z]+", c[0].lower()))
            new_text.append(temp)
        return []
    
    
    else:
        temp_list = names
        return set(names)


def fuzzy_matching(text):
    global new_text
    # print(text)
    #Declarations

    # if blur_stat == 0:

    Fname_ds = []
    Lname_ds = []
    body_part = ['abdomen', 'barium', 'bone', 'chest', 'dental', 'extremity', 'hand', 'joint', 'neck', 'pelvis', 'sinus', 'skull', 'spine', 'thoracic']

    print(text)

    #Opening CSV files
    with open('Indian_Names.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for e in reader:
            Fname_ds.append(e[1])

    with open('indian_last_name.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader)

        for e in reader:
            Lname_ds.append(e[0])


    if len(text) < 3 or pr.extractOne(text, body_part)[1] > 75: 
        return 

    if text.isalnum(): 
        new_text.append(text)
        return

    #Checking for first name
    for given in Fname_ds:
        if fz.ratio(text, given) > 60:
            # print(text, given, 1)
            new_text.append(text)
            return

    #Checking for last name
    for given in Lname_ds:
        if fz.ratio(text, given) > 60:
            # print(text, given, 2)
            new_text.append(text)
            return

    # blur_stat = 0



def auto_crop(img):
    

    aspect_ratio = img.shape[1]/img.shape[0]
    height = int(1500/aspect_ratio)
    img = cv2.resize(img, (1500, height))

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
        
        print(len(ROI))
        return(warped)
    
    return(img)
    