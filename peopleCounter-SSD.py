# Allows the use of the opencv package through the reference "cv"
import cv2 as cv

from trackableobject import TrackableObject
from tracker import *

MAX_PEOPLE = 5
DOOR_LIMIT = 200

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
totalUp = 0
totalDown = 0
trackableObjects = {}
while True:
    ret, frame = capDevice.read()

    # Obtains the class IDs, confidence values and bounding boxes from the image
    classIDs, confidence, bbox = network.detect(frame,confThreshold=0.4)
    detections = []

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
                detections.append([x, y, w, h])
                tracker = EuclideanDistTracker()
                boxes_ids = tracker.update(detections)

                for box_id in boxes_ids:
                    x, y, w, h, objId = box_id
                    to = trackableObjects.get(objId, None)

                    # if there is no existing trackable object, create one
                    if to is None:
                        to = TrackableObject(objId, box_id)

                    if y > to.bbox[1] :
                        direction = 1 # UPPER
                    else:
                        direction = -1 # DOWN

                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction < 0 and y < DOOR_LIMIT:
                            totalUp += 1
                            to.counted = True

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif direction > 0 and y > DOOR_LIMIT:
                            totalDown += 1
                            to.counted = True




                # Draws a bounding box and writes the class name of the object identified
                totalPeople = totalPeople + 1
                if totalPeople > MAX_PEOPLE:
                    rectColor = (0, 0, 255)
                else:
                    rectColor = (0, 255, 0)

                cv.rectangle(frame, (x, y), (x + w, y + h), rectColor, 5)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Up", totalUp),
            ("Down", totalDown)
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv.putText(frame, text, (10, 10 - ((i * 20) + 20)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv.imshow("Frame", frame)
    if cv.waitKey(1) == ord("q"):
        break


capDevice.release()
cv.destroyAllWindows()
