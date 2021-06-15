# Allows the use of the opencv package through the reference "cv"
import cv2 as cv

# Setup a video capture device. 0 is usually the inbuilt webcam
capDevice = cv.VideoCapture(0, cv.CAP_DSHOW)

# Setup the Haar Cascades for the different objects
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
#https://github.com/opencv/opencv/tree/master/data/haarcascades
#upperBody = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_upperBody.xml")


while True:
    ret, frame = capDevice.read()

    # Convert to image to gray and run the face matching
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3,5)

    for (x, y, w, h) in faces:
        # Draws rectangles around the identified faces
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        roiGray = gray[y:y+h, x+w]
        roiColour = frame[y:y+h, x:x+w]

        # Runs the eye matching
        eyes = eyeCascade.detectMultiScale(gray, 1.3, 5)
        for (xe, ye, we, he) in eyes:
            # Draws rectangles around the identified faces
            cv.rectangle(frame, (xe, ye), (xe + we, ye + he), (255, 0, 0), 5)
            roiGray = gray[y:ye + he, xe + we]
            roiColour = frame[y:ye + he, x:xe + we]

    cv.imshow("Frame", frame)

    if cv.waitKey(1) == ord("q"):
        break


# Releases the video capture device and closes all windows
capDevice.release()
cv.destroyAllWindows()
