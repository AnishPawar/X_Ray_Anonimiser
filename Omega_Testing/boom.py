import cv2
import numpy as np 
# cap = cv2.VideoCapture(0)

def mouse(event,x,y,z,w):
    print(x,y)
    pixel = frame[y,x]
    print(pixel)
    

#Mouse function
cv2.namedWindow('frame')
cv2.setMouseCallback('frame',mouse)

   
while True:
    frame = cv2.imread("credimages/test33.png")
    # ret, frame = cap.read()     
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()