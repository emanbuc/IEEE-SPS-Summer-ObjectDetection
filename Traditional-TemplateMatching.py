# Allows the use of the opencv package through the reference "cv"
import cv2 as cv

# Reads the template object and the image to search for the object
imgTemplate = cv.imread("Files/template.png", 0)
img = cv.imread("Files/image_templatematching.png", 0)

# Resizes the images to fit on screen
imgTemplate = cv.resize(imgTemplate, (0, 0), fx=0.5, fy=0.5)
img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)

# Obtains the size of the template image
height, width = imgTemplate.shape

# List of different template matching methods
templateMethods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR,
                   cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]


for method in templateMethods:
    img2 = img.copy()

    # Matches the two images for a particular method
    result = cv.matchTemplate(img, imgTemplate, method)
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)



    # Some of the methods require the minimum location value while others require the maximum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        location = minLoc
    else:
        location = maxLoc

    # Identify the location of the object on the image
    bottomRight = (location[0]+width, location[1]+height)


    # Draws a rectangle over the template object in the image
    imgRectangle = cv.rectangle(img2, location, bottomRight, 255, 3)
    windowName = "matchmode "+str(templateMethods[method])


    cv.imshow(windowName, img2)

    cv.waitKey(0)
    cv.destroyAllWindows()