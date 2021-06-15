# Allows the use of the opencv package through the reference "cv"
import cv2 as cv

# Reads the template object and the image to search for the object
imgTemplate = cv.imread("Files/template2.png", 0)
#img = cv.imread("Files/image_orb.png", 0)
img = cv.imread("Files/template2.png", 0)

# Resizes the image
imgTemplate = cv.resize(imgTemplate, (0, 0), fx=1, fy=1)
img = cv.resize(img, (0, 0), fx=1, fy=1)

# Create an ORB object
orb = cv.ORB_create(nfeatures=1000)

# Creates the key points for the images
kpImg, desImg = orb.detectAndCompute(img, None)
kpImgTemplate, desImgTemplate = orb.detectAndCompute(imgTemplate, None)

# Draws the key points of the images
# imgKp = cv.drawKeypoints(img, kpImg, None)
# imgTemplateKp = cv.drawKeypoints(imgTemplate, kpImgTemplate, None)

# cv.imshow("KP Image", imgKp)
# cv.imshow("KP ImageTemplate", imgTemplateKp)

# Creates a brute force matches and uses KNN to do the matching
bf = cv.BFMatcher()
matches = bf.knnMatch(desImgTemplate, desImg, k=2)


# Creates a list of good matches based on the distance
goodMatches = []
for x, y in matches:
    if x.distance < 0.75*y.distance:
        goodMatches.append([x])


# Draws the matches between the two images
imgMatch= cv.drawMatchesKnn(img, kpImg, imgTemplate, kpImgTemplate, goodMatches, None, flags=2)


cv.imshow("Image Matches", imgMatch)
print("Number of good matches: " + str(len(goodMatches)))

# Waits for the user to press a key
cv.waitKey(0)
cv.destroyAllWindows()
