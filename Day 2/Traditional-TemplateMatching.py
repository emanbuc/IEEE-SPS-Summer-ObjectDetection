# Allows the use of the opencv package through the reference "cv"
import

# Reads the template object and the image to search for the object
imgTemplate = 
img = 

# Resizes the images to fit on screen
imgTemplate = 
img = 

# Obtains the size of the template image
height, width = imgTemplate.shape

# List of different template matching methods
templateMethods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR,
                   cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]


for method in templateMethods:
    img2 = img.copy()

    # Matches the two images for a particular method


    # Some of the methods require the minimum location value while others require the maximum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        location = minLoc
    else:
        location = maxLoc

    # Identify the location of the object on the image


    # Draws a rectangle over the template object in the image


    cv.imshow(windowName, img2)

    cv.waitKey(0)
    cv.destroyAllWindows()