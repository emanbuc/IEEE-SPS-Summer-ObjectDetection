import cv2 as cv
import numpy as np

capDevice = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = capDevice.read()

    width = int(capDevice.get(3))
    height = int(capDevice.get(3))

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lowerBlue= np.array([90,50,50])
    upperBlue = np.array([130, 255, 255])

    # create a blue mask to show only blue
    mask= cv.inRange(hsv, lowerBlue,upperBlue)

    result = cv.bitwise_and(frame,frame, mask=mask)
    cv.imshow("webcam image",frame)
    cv.imshow("image with color mask", result)

    #some overlay on captured video frame
    imgLine=cv.line(frame, (0, 0), (width,height), (255, 255, 255), 5)
    imgRectangle = cv.rectangle(frame, (100, 100), (200, 200), (0, 0, 0), 5)
    imgCircle = cv.circle(frame, (250, 250), 50, (255, 0, 0), 5)

    font = cv.FONT_HERSHEY_SIMPLEX
    imgText = cv.putText(frame, "Text here", (200, height-5), font, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow("Webcam Frame", frame)

    if cv.waitKey(1) == ord('q'):
        break

# relesae video capture
capDevice.release()
cv.destroyAllWindows()
