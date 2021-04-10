import imutils
import cv2
net = cv2.dnn.readNet("/Users/anishpawar/GID_9_2021/X_Ray_Anonimiser/Omega_Testing/frozen_east_text_detection.pb")

frame = cv2.imread('credimages/visa.png')
frame = cv2.resize(frame,(320,320))

blob = cv2.dnn.blobFromImage(frame, 1.0, (320,320), (123.68, 116.78, 103.94), True, False)

outputLayers = []
outputLayers.append("feature_fusion/Conv_7/Sigmoid")
outputLayers.append("feature_fusion/concat_3")


net.setInput(blob)
output = net.forward(outputLayers)
scores = output[0]
geometry = output[1]

print(output)

[boxes, confidences] = decode(scores, geometry, 0.6)

# indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.6, nmsThreshold)
