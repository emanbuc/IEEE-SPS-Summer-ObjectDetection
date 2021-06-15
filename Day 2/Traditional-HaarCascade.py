# Allows the use of the opencv package through the reference "cv"
import

# Setup a video capture device. 0 is usually the inbuilt webcam
capDevice = cv.VideoCapture(0, cv.CAP_DSHOW)

# Setup the Haar Cascades for the different objects
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")



while True:
    ret, frame = capDevice.read()

    # Convert to image to gray and run the face matching
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    for (x, y, w, h) in faces:

        # Draws rectangles around the identified faces


        # Runs the eye matching


    cv.imshow("Frame", frame)

    if cv.waitKey(1) == ord("q"):
        break


# Releases the video capture device and closes all windows
capDevice.release()
cv.destroyAllWindows()
