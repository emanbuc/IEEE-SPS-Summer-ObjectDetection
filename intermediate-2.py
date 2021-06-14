import cv2 as cv

capDevice = cv.VideoCapture(0,cv.CAP_DSHOW)

while True:
    ret, frame = capDevice.read()

    width = int(capDevice.get(3))
    height = int(capDevice.get(3))

    imgLine=cv.line(frame, (0, 0), (width,height), (255, 255, 255), 5)
    imgRectangle = cv.rectangle(frame, (100, 100), (200,200), (0, 0, 0), 5)
    imgCircle = cv.circle(frame, (250, 250), 50, (255, 0, 0), 5)

    font = cv.FONT_HERSHEY_SIMPLEX
    imgText = cv.putText(frame, "Text here", (200, height-5), font, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow("Webcam Frame", frame)

    if cv.waitKey(1) == ord('q'):
        break

# relesae video capture
capDevice.release()
cv.destroyAllWindows()