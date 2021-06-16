# Allows the use of the opencv package through the reference "cv"
import cv2 as cv
import numpy as np

size = 320
confThreshold = 0.5
nmsThreshold = 0.3

# Setup a video capture device. 0 is usually the inbuilt webcam
capDevice = capDevice = cv.VideoCapture(0, cv.CAP_DSHOW)

# Defines the classes file used in YOLO
classNamesFile = "Files/coco.names"

# Reads the classes file and stores them in classNames
with open(classNamesFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")


# Defines the YOLO configuration and weights files
modelConfig = "Files/yolov3.cfg"
modelWeights = "Files/yolov3.weights"

# Sets the YOLO configuration and weights for the network
network = cv.dnn.readNetFromDarknet(modelConfig, modelWeights)
network.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
network.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)



def findObjects(outputs, img):
    height, width, channel = img.shape
    bbox = []
    classIDs = []
    confidence = []

    # Obtains the objects that are greater than the confidence threshold defined
    for output in outputs:
        for detection in output:
            scores=detection[5:]
            classID=np.argmax(scores)
            conf = scores[classID]
            if conf > confThreshold:
                w,h = int(detection[2]*width), int(detection[3]*height)
                x,y = int((detection[0]*width)-w/2), int((detection[1]*height)-h/2)
                bbox.append([x,y,w,h])

                classIDs.append(classID)
                confidence.append(float(conf))

    # Obtains the unique bounding area for objects identified
    indices = cv.dnn.NMSBoxes(bbox, confidence, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        # Draws a bounding box and writes the class name of the object identified
        cv.rectangle(img, (x, y), (x+w, y+h), (255,0,0) , 2)
        cv.putText(img, f"{classNames[classIDs[i]].upper()} {int(confidence[i]*100)}%",
                   (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)


while True:
    ret, frame = capDevice.read()

    # Creates a blob from the image and sets it as input to the network
    blob = cv.dnn.blobFromImage(frame,1/255,(size,size), (8,0,0),1, crop=False)
    network.setInput(blob)

    # Identifies the output layers and obtains the output from the network
    layerNames = network.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in network.getUnconnectedOutLayers()]
    netOutputs = network.forward(outputNames)

    # Finds the objects in the image
    findObjects(netOutputs, frame)
    cv.imshow("Frame", frame)

    if cv.waitKey(1) == ord("q"):
        break


# Releases the video capture device and closes all windows
capDevice.release()
cv.destroyAllWindows()
