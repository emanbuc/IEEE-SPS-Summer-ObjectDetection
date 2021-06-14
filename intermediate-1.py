import cv2 as cv

capDevice = cv.VideoCapture(0,cv.CAP_DSHOW)
capDevice = cv.VideoCapture(0)

while True:
    ret, frame = capDevice.read()
    cv.imshow("Webcam Frame", frame)

    if cv.waitKey(1) == ord('q'):
        break

#relesae video capture
capDevice.release()
cv.destroyAllWindows()