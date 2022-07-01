#importing the module cv2
import cv2

#reading the image whose shape is to be detected using imread() function
imageread = cv2.imread('maps/mymap.png')

#converting the input image to grayscale image using cvtColor() function
imagegray = cv2.cvtColor(imageread, cv2.COLOR_BGR2GRAY)

#using threshold() function to convert the grayscale image to binary image
_, imagethreshold = cv2.threshold(imagegray, 245, 255, cv2.THRESH_BINARY_INV)

#finding the contours in the given image using findContours() function
imagecontours, _ = cv2.findContours(imagethreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#for each of the contours detected, the shape of the contours is approximated using approxPolyDP() function and the contours are drawn in the image using drawContours() function
for count in imagecontours:
    epsilon = 0.01 * cv2.arcLength(count, True)
    approximations = cv2.approxPolyDP(count, epsilon, True)
    cv2.drawContours(imageread, [approximations], 0, (0), 3)

#displaying the resulting image as the output on the screen
cv2.imshow("Resulting_image", imageread)
cv2.waitKey(0)