# Allows the use of the opencv package through the reference "cv"
import 

# Create an ORB object
orb = 

# Reads the template object
# Resizes the image
# Creates the key points for the image
imgTemplate = 
imgTemplate = 
kpImgTemplate, desImgTemplate = 

# Setup a video capture device. 0 is usually the inbuilt webcam
capDevice = cv.VideoCapture(0, cv.CAP_DSHOW)

# Sets threshold of number of matches to identify an object
threshold = 10


def findMatches(img):
    imgOrg = img.copy()

    # Convert to image to gray
    img = 

    # Creates the key points for the images
    kpImg, desImg = 

    # Creates a brute force matches and uses KNN to do the matching


    # Creates a list of good matches based on the distance
    numMatches = []
    try:
        for x, y in matches:

        # Draws the matches between the two images
        
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
        print("Number of good matches: " + )
        cv.imshow("Image Found", imgText)

    if cv.waitKey(1) == ord("q"):
        break


# Releases the video capture device and closes all windows
capDevice.release()
cv.destroyAllWindows()
