# Allows the use of the opencv package through the reference "cv"
import cv2 as cv

# Create an ORB object
orb = cv.ORB_create(nfeatures=1000)

# Reads the template object
# Resizes the image
# Creates the key points for the image
imgTemplate = cv.imread("Files/template2.png", 0)
imgTemplate = cv.resize(imgTemplate, (0, 0), fx=1, fy=1)
kpImgTemplate, desImgTemplate = orb.detectAndCompute(imgTemplate, None)

# Setup a video capture device. 0 is usually the inbuilt webcam
capDevice = cv.VideoCapture(0, cv.CAP_DSHOW)

# Sets threshold of number of matches to identify an object
threshold = 10


def findMatches(img):
    imgOrg = img.copy()

    # Convert to image to gray
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Creates the key points for the images
    kpImg, desImg = orb.detectAndCompute(img,None)

    # Creates a brute force matches and uses KNN to do the matching
    bf = cv.BFMatcher()
    matches = bf.knnMatch(desImgTemplate, desImg, k=2)


    # Creates a list of good matches based on the distance
    numMatches = []
    try:
        for x, y in matches:
            if x.distance < 0.75 * y.distance:
                goodMatches.append([x])

        # Draws the matches between the two images
        imgMatch = cv.drawMatchesKnn(img, kpImg, imgTemplate, kpImgTemplate, goodMatches, None, flags=2)

        cv.imshow("Image Matches", imgMatch)
    except:
        pass
    return numMatches


while True:
    ret, frame = capDevice.read()

    goodMatches = findMatches(frame)

    if len(goodMatches) > threshold:
        # Writes text
        font = cv.FONT_HERSHEY_SIMPLEX
        imgText = cv.putText(frame, "Image Found", (10, 20), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        print("Number of good matches: " + str(len(goodMatches)))
        cv.imshow("Image Found", imgText)

    if cv.waitKey(1) == ord("q"):
        break


# Releases the video capture device and closes all windows
capDevice.release()
cv.destroyAllWindows()
