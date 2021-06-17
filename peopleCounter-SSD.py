# Allows the use of the opencv package through the reference "cv"
import cv2 as cv
import imutils as imutils

from trackableobject import TrackableObject
from tracker import *

MAX_PEOPLE = 5
ZONE_FENCH=(10,100,300,600)
TRACKING_FRAME_NUMBER = 10
COLOR_OK = (0, 255, 0) # GREEN
COLOR_INFO = (255, 255, 255) # WHITE
COLOR_ALERT = (0, 0, 255) #RED

def isInside(x,y):
    inside = (ZONE_FENCH[0]<x) & (x<ZONE_FENCH[2]) & (ZONE_FENCH[1]<y) &(y<ZONE_FENCH[3])
    return inside

# Setup a video capture device. 0 is usually the inbuilt webcam
# capDevice = capDevice = cv.VideoCapture(0, cv.CAP_DSHOW)
capDevice = capDevice = cv.VideoCapture("Files/door_entrance2.mp4")

# Defines the classes file used in YOLO
classNamesFile = "Files/coco.names"

# Reads the classes file and stores them in classNames
with open(classNamesFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Defines the SSD configuration and weights files
modelConfig = "Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
modelWeights = "Files/frozen_inference_graph.pb"

# Sets up the SSD with various settings
network = cv.dnn_DetectionModel(modelConfig, modelWeights)
network.setInputSize(320, 320)
network.setInputScale(1.0 / 127.5)
network.setInputMean((127.5, 127.5, 127.5))
network.setInputSwapRB(True)


totalPeopleInside = 0 # People inside the monitored area detected

trackedObjects = {}
H = None
W = None
tracker = EuclideanDistTracker()

while True:
    totalPeopleInside=0
    ret, frame = capDevice.read()

    if frame is None:
        break

    # resize the frame (the less data we have, the faster we can process it)
    #frame = imutils.resize(frame, width=500)

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Obtains the class IDs, confidence values and bounding boxes from the image
    classIDs, confidence, bbox = network.detect(frame, confThreshold=0.4)
    detections = []

    # Draws a bounding box and writes text only if a class has been identified
    if len(classIDs) != 0:
        for classID, conf, box in zip(classIDs.flatten(), confidence.flatten(), bbox):
            #print(classID)
            # Writes the class name and confidence for classes in the COCO file
            if classID == 1: # only People
                x, y, w, h = box[0], box[1], box[2], box[3]

                cv.putText(frame, f"{int(conf * 100)}%", (box[0] + 150, box[1] + 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                detections.append([x, y, w, h])

        boxes_ids = tracker.update(detections)

        for box_id in boxes_ids:
            bx, by, bw, bh, objId = box_id
            cy = (by + by + bh) // 2
            cx = (bx + bx + bw) // 2

            to = trackedObjects.get(objId, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objId, box_id)
                trackedObjects[objId] = to

            if isInside(cx,cy):
                totalPeopleInside = totalPeopleInside + 1

            rectColor = COLOR_INFO

            if (totalPeopleInside > MAX_PEOPLE) & isInside(cx,cy):
                rectColor = COLOR_ALERT
            elif isInside(cx,cy):
                rectColor = COLOR_OK

            cv.circle(frame, (cx, cy), 4, COLOR_INFO, -1)
            cv.putText(frame, "ID: "+str(to.objectID) +"( "+str(cx)+","+str(cy)+")", (bx + 10, by + 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_INFO, 2)
            cv.rectangle(frame, (bx, by), (bx + bw, by + bh), rectColor, 2,-1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
           # ("Move Up", totalUp),
           # ("Move Down", totalDown),
            ("People inside", totalPeopleInside),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv.putText(frame, text, (10, H - ((i * 20) + 20)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, rectColor, 2)

    cv.rectangle(frame,ZONE_FENCH, COLOR_INFO, 3)
    #cv.line(frame, (0, ZONE_LIMIT_Y), (W, ZONE_LIMIT_Y), (255, 255, 255), 3)
    cv.imshow("Frame", frame)
    if cv.waitKey(1) == ord("q"):
        break

capDevice.release()
cv.destroyAllWindows()
