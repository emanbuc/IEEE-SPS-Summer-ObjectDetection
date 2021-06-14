import cv2 as cv
img = cv.imread("Files/basic.png", 0)

img = cv.resize(img, (0, 0), fx=2, fy=2)
imgRotated = img.copy()
imgRotated = cv.rotate(imgRotated, cv.ROTATE_90_CLOCKWISE)
cv.imshow("Image", img)
cv.imshow("Image Rotated", imgRotated)

cv.waitKey(0)
cv.destroyAllWindows()

print("this value", cv.ROTATE_90_CLOCKWISE)
