# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi


import argparse
from datetime import time

import cv2 as cv
import imutils as imutils
from imutils.video import FPS

from point import Point
from trackableobject import TrackableObject
from tracker import *

# ===============================================================
# Global variables
MAX_PEOPLE = 5
ZONE_FENCH = (100, 100, 800, 500)
TRACKING_FRAME_NUMBER = 10
COLOR_OK = (0, 255, 0)  # GREEN
COLOR_INFO = (255, 255, 255)  # WHITE
COLOR_ALERT = (0, 0, 255)  # RED
#
# --------------------------------------------------------------

# ===============================================================
# construct the argument parse and parse the arguments
# --------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-r", "--resize", type=int, default=600,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


#
# --------------------------------------------------------------


# ===============================================================
# UTILITY FUNCTIONS
# --------------------------------------------------------------

def is_inside(point: Point):
    inside = (ZONE_FENCH[0] < point.x) & (point.x < ZONE_FENCH[2]) \
             & (ZONE_FENCH[1] < point.y) & (point.y < ZONE_FENCH[3])
    return inside


def init_video_capture_device():
    device = None
    # if a video path was not supplied, grab a reference to the webcam
    if not args.get("input", False):
        print("[INFO] starting video stream...")
        # Setup a video capture device. 0 is usually the inbuilt webcam
        device = cv.VideoCapture(0, cv.CAP_DSHOW)
        time.sleep(2.0)

    # otherwise, grab a reference to the video file
    else:
        print("[INFO] opening video file...")
        device = cv.VideoCapture(args["input"])
    return device


#
# --------------------------------------------------------------


# ==============================================================
#  ************* MAIN ******************************************
# ---------------------------------------------------------------
capDevice = init_video_capture_device()
if capDevice is None:
    print("Video Capture Device is NONE")

print("press 'q' key to EXIT")
# Defines the SSD configuration and weights files
modelConfig = args["prototxt"]
modelWeights = args["model"]

totalPeopleInside = 0  # People inside the monitored area detected

trackedObjects = {}
H = None
W = None
# initialize the video writer (we'll instantiate later if need be)
writer = None

#ret, frame = capDevice.read()
#frame = imutils.resize(frame, width=args["resize"])
#(H, W) = frame.shape[:2]

print("resize to width: " + str(args["resize"]))
print("Frame size: W: " + str(W) + " H: " + str(H))

# Sets up the SSD with various settings
network = cv.dnn_DetectionModel(modelConfig, modelWeights)
#network.setInputSize(W, H)
network.setInputSize(320, 320)
network.setInputScale(1.0 / 127.5)
network.setInputMean((127.5, 127.5, 127.5))
network.setInputSwapRB(True)

tracker = EuclideanDistTracker()

# start the frames per second throughput estimator
fps = FPS().start()

while True:
    totalPeopleInside = 0
    ret, frame = capDevice.read()

    if frame is None:
        break

    # resize the frame (the less data we have, the faster we can process it)
    #frame = imutils.resize(frame, width=args["resize"])

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # Obtains the class IDs, confidence values and bounding boxes from the image
    classIDs, confidence, bbox = network.detect(frame, args["confidence"])
    detections = []

    # Draws a bounding box and writes text only if a class has been identified
    if len(classIDs) != 0:
        for classID, conf, box in zip(classIDs.flatten(), confidence.flatten(), bbox):
            # print(classID)
            # Writes the class name and confidence for classes in the COCO file
            if classID == 1:  # only People
                x, y, w, h = box[0], box[1], box[2], box[3]

                cv.putText(frame, f"{int(conf * 100)}%", (box[0] + 150, box[1] + 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                detections.append([x, y, w, h])

        boxes_ids = tracker.update(detections)

        for box_id in boxes_ids:
            bx, by, bw, bh, objId = box_id
            centroid = Point((bx + bx + bw) // 2, (by + by + bh) // 2)

            to = trackedObjects.get(objId, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objId, box_id)
                trackedObjects[objId] = to

            if is_inside(centroid):
                totalPeopleInside = totalPeopleInside + 1

            rectColor: tuple[int, int, int] = COLOR_INFO

            if (totalPeopleInside > MAX_PEOPLE) & is_inside(centroid):
                rectColor = COLOR_ALERT
            elif is_inside(centroid):
                rectColor = COLOR_OK

            cv.circle(frame, (centroid.x, centroid.y), 4, rectColor, -1)
            cv.putText(frame, "ID: " + str(to.objectID), (bx + 10, by + 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_INFO, 2)
            cv.rectangle(frame, (bx, by), (bx + bw, by + bh), rectColor, 2, -1)

    cv.putText(frame, "People inside: " + str(totalPeopleInside), (10, H - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, rectColor, 2)

    # Draw Fench overlay
    cv.rectangle(frame, ZONE_FENCH, (100,100,100), 2, -1)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    cv.imshow("Frame", frame)
    if cv.waitKey(1) == ord("q"):
        break

    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

capDevice.release()
cv.destroyAllWindows()
