# Allows the use of the opencv package through the reference "cv"
import

# Reads the template object and the image to search for the object
imgTemplate = 

# Resizes the image
imgTemplate = 
img = 

# Create an ORB object


# Creates the key points for the images


# Draws the key points of the images
# imgKp = cv.drawKeypoints(img, kpImg, None)
# imgTemplateKp = cv.drawKeypoints(imgTemplate, kpImgTemplate, None)

# cv.imshow("KP Image", imgKp)
# cv.imshow("KP ImageTemplate", imgTemplateKp)

# Creates a brute force matches and uses KNN to do the matching


# Creates a list of good matches based on the distance
goodMatches = []
for x, y in matches:


# Draws the matches between the two images


cv.imshow("Image Matches", imgMatch)
print("Number of good matches: " + str(len(goodMatches)))

# Waits for the user to press a key
cv.waitKey(0)
cv.destroyAllWindows()
