import cv2 as cv
import numpy as np
import sys
import os
import wget

img_path = ".\\static\\test.jpg"
output_path = ".\\static\\result.jpg"

modelSize = 608
modelConfig = "yolov3.cfg"
modelWeights = "yolov3.weights"

if not os.path.isfile(modelWeights):
    wget.download('https://pjreddie.com/media/files/yolov3.weights')
if not os.path.isfile(modelConfig):
    wget.download('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg')

confidenceThreshold = 0.9
nmsThreshold = 0.1

net = cv.dnn.readNetFromDarknet(modelConfig,modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

classNames = []
with open('coco.names','rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

def preProcessImage(img):
    blob = cv.dnn.blobFromImage(img, 1/255, (modelSize, modelSize), [0,0,0], 1, crop = False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    return outputs

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confidences = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([w,h,x,y])
                classIds.append(classId)
                confidences.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confidences, confidenceThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        w,h,x,y = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)
        cv.putText(img, f'{classNames[classIds[i]]}: {int(confidences[i]*100)}%', (max(x,0),y-10), cv.FONT_HERSHEY_COMPLEX, 0.6, (255,0,255),2)

def detect():
    img = cv.imread(cv.samples.findFile(img_path))

    if img is None:
        sys.exit("Could not read the image.")

    findObjects(preProcessImage(img), img)

    cv.imwrite(output_path, img)

if __name__ == "__main__":
    img = cv.imread(cv.samples.findFile(img_path))

    if img is None:
        sys.exit("Could not read the image.")

    #img = cv.resize(img, (modelSize,modelSize))

    findObjects(preProcessImage(img), img)

    cv.imshow("Display window", img)
    k = cv.waitKey(0)