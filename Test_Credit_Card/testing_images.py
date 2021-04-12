from tesseract_function import tesseacr_func
from preprocessing import preprocess
from imutils import contours
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import pytesseract


# Digit Classifier
def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    return Y

from keras.models import Sequential
from keras.models import model_from_json

def run_preds(img):
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
    
    model_file = open('/Users/anishpawar/Downloads/Credit-Card-Number-Recognition-master-2/Digit-Classifier-master/Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("/Users/anishpawar/Downloads/Credit-Card-Number-Recognition-master-2/Digit-Classifier-master/Data/Model/weights.h5")
    op = predict(model, img)
    return op
    


ref = cv2.imread("Images/ocr_a_reference.png")
ref1 = cv2.imread("FONTS/OCR-A_2.png")


ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
ref1 = cv2.cvtColor(ref1, cv2.COLOR_BGR2GRAY)
ref1 = cv2.threshold(ref1, 10, 255, cv2.THRESH_BINARY_INV)[1]

refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
refCnts = refCnts[1] if imutils.is_cv3() else refCnts[0]
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

refCnts1 = cv2.findContours(ref1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
refCnts1 = refCnts1[1] if imutils.is_cv3() else refCnts1[0]
refCnts1 = contours.sort_contours(refCnts1, method="left-to-right")[0]


digits = {}
i = 0
for c in refCnts:
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:(y + h), x:(x + w)]
    roi2 = cv2.resize(roi, (57, 88))
    print(digits)
    digits[i] = roi
    i+=1
 




image = cv2.imread("/Users/anishpawar/GID_9_2021/X_Ray_Anonimiser/Test_Credit_Card/Images/test33.png")
org = image.copy()

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray_og = gray.copy()
image = imutils.resize(image, width=300)

aspect_ratio = org.shape[0]/image.shape[0]
gray,thresh = preprocess(image)


cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv3() else cnts[0]


locs = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    # ar = w / float(h)
    # if  ar > 2.5 and ar < 4.5:
        # if  (int(w*aspect_ratio) > int(40*aspect_ratio) and int(w*aspect_ratio) < int(70*aspect_ratio)) and (int(h*aspect_ratio) > int(10*aspect_ratio) and int(h*aspect_ratio) < int(20*aspect_ratio)):
    locs.append((x, y, w, h))


results = []
kernel = np.ones((7,7))
thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv3() else cnts[0]
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)

    x =tesseacr_func(x,y,w,h,aspect_ratio,org)
    print(x[0])
    start_X = x[0]
    start_Y = x[1]
    end_X = x[2]
    end_Y =x[3]
    text= x[4]
    # for start_X, start_Y, end_X, end_Y, text in x :

    cv2.rectangle(org, (start_X, start_Y), (end_X, end_Y),
        (0, 0, 255), 2)
    cv2.putText(org, text, (start_X, start_Y),
        cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0, 255), 3)

    results.append(x)
print(results)

	

locs = sorted(locs, key=lambda x:x[0])  
locs
output = []
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    print(i,(gX, gY, gW, gH))

for (gX, gY, gW, gH) in locs:
    print((gX, gY, gW, gH))
    groupOutput = []
    group = gray_og[int((gY)*aspect_ratio) :int((gY + gH)*aspect_ratio) , int((gX)*aspect_ratio) :int((gX + gW)*aspect_ratio)]
    
    image_area = org.shape[0]*org.shape[1]
    group_area = group.shape[0]*group.shape[1]

    Y1 = int((gY)*aspect_ratio)
    Y2 = int((gY + gH)*aspect_ratio)
    X1 = int((gX)*aspect_ratio)
    X2 = int((gX + gW)*aspect_ratio)


    if (int(org.shape[1]*0.3))<Y1<(int(org.shape[1]*0.5)) and (int(org.shape[1]*0.3))<Y2<(int(org.shape[1]*0.5)) and (group_area>= 0.01*image_area):


        width = group.shape[1]
        digitsk = [group[:,0:int(width*0.25)],group[:,int(width*0.25):int(width*0.5)],group[:,int(width*0.5):int(width*0.75)],group[:,int(width*0.75):]]

        for c in digitsk:
            if len(c)>3:
                
                roi = cv2.resize(c, (54, 85))
                # plt.imshow(roi)
                op = run_preds(roi)

                test = cv2.resize(roi,(28,28))
                cv2.imwrite("Test.jpg",test)

            scores = []
            for (digit, digitROI) in digits.items():
                result = cv2.matchTemplate(roi, digitROI,
                                        cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)

                scores.append(score)
            
            groupOutput.append(str(op[0]))
            cv2.rectangle(image, (gX - 5, gY - 5),
                        (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
            cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1) 
        output.extend(groupOutput)

print("Credit Card #: {}".format("".join(output)))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(org, cv2.COLOR_BGR2RGB))
plt.show()