# Allows the use of the opencv package through the reference "cv"
import
import 

# Define variables
size = 320
confThreshold = 0.5
nmsThreshold = 0.3

# Setup a video capture device. 0 is usually the inbuilt webcam
capDevice = 

# Defines the classes file used in YOLO
classNamesFile = "Files/coco.names"

# Reads the classes file and stores them in classNames


# Defines the YOLO configuration and weights files
modelConfig = "Files/yolov3.cfg"
modelWeights = "Files/yolov3.weights"

# Sets the YOLO configuration and weights for the network


def findObjects(outputs, img):
    height, width, channel = 
    bbox = []
    classIDs = []
    confidence = []

    # Obtains the objects that are greater than the confidence threshold defined
    for output in outputs:
        for detection in output:


            if conf > confThreshold:


                classIDs.append(classID)
                confidence.append(float(conf))

    # Obtains the unique bounding area for objects identified
    indices = cv.dnn.NMSBoxes(bbox, confidence, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = 
        x, y, w, h = 

        # Draws a bounding box and writes the class name of the object identified
        cv.rectangle()
        cv.putText(img, f"{classNames[classIDs[i]].upper()} {int(confidence[i]*100)}%",
                   (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)


while True:
    ret, frame = capDevice.read()

    # Creates a blob from the image and sets it as input to the network
    blob = 
    network.setInput(blob)

    # Identifies the output layers and obtains the output from the network
    layerNames = 
    outputNames = 
    netOutputs = 

    # Finds the objects in the image
    findObjects(netOutputs, frame)
    cv.imshow("Frame", frame)

    if cv.waitKey(1) == ord("q"):
        break


# Releases the video capture device and closes all windows
capDevice.release()
cv.destroyAllWindows()
