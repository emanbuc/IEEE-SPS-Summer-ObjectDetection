# Allows the use of the opencv package through the reference "cv"
import cv2 as cv
import numpy as np

MAX_PEOPLE = 5

# Setup a video capture device. 0 is usually the inbuilt webcam
#capDevice = capDevice = cv.VideoCapture(0, cv.CAP_DSHOW)
capDevice = capDevice = cv.VideoCapture("Files/shop.mp4")

# Defines the classes file used in YOLO
classNamesFile = "Files/coco.names"

# Reads the classes file and stores them in classNames
with open(classNamesFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")


# Defines the SSD configuration and weights files
modelConfig = "Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
modelWeights = "Files/frozen_inference_graph.pb"

# Sets up the SSD with various settings
network = cv.dnn_DetectionModel(modelConfig,modelWeights)
network.setInputSize(320, 320)
network.setInputScale(1.0/127.5)
network.setInputMean((127.5, 127.5, 127.5))
network.setInputSwapRB(True)

totalPeople= 0
while True:
    totalPeople = 0
    ret, frame = capDevice.read()

    # Obtains the class IDs, confidence values and bounding boxes from the image
    classIDs, confidence, bbox = network.detect(frame,confThreshold=0.5)

    # Draws a bounding box and writes text only if a class has been identified
    if len(classIDs)!=0:
        for classID, conf, box in zip(classIDs.flatten(), confidence.flatten(), bbox):
            print(classID)
            # Writes the class name and confidence for classes in the COCO file
            if classID== 1:
                cv.putText(frame, classNames[classID-1].upper(), (box[0]+10, box[1]+30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(frame, f"{int(conf*100)}%", (box[0] + 150, box[1] + 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                x, y, w, h = box[0], box[1], box[2], box[3]

                # Draws a bounding box and writes the class name of the object identified
                totalPeople = totalPeople + 1
                if totalPeople > MAX_PEOPLE:
                    rectColor = (0, 0, 255)
                else:
                    rectColor = (0, 255, 0)

                cv.rectangle(frame, (x, y), (x + w, y + h), rectColor, 5)

    cv.imshow("Frame", frame)
    if cv.waitKey(1) == ord("q"):
        break


capDevice.release()
cv.destroyAllWindows()
