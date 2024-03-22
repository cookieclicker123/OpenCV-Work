import cv2 as cv
import numpy as np

# Read the image
img = cv.imread('images/baseball.jpg')

# Assuming a screen resolution of 1920x1080 (you should adjust this to your actual screen resolution)
#screen_res = (2560, 1440)
#img_height, img_width = img.shape[:2]

# Calculate the position to center the window
#position_x = (screen_res[0] - img_width) // 2
#position_y = (screen_res[1] - img_height) // 2

# Create a window that can be resized (allowing for the window to be moved)
#cv.namedWindow('image', cv.WINDOW_NORMAL)

# Move the window to the center of the screen
#cv.moveWindow('image', position_x, position_y)

# Show the image


def translate(img, x,y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x --> Left
# -y --> Up
# x --> Right
# y --> Down

translated = translate(img, 100, 100) 


def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)


rotated = rotate(img, -90)

resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)

flip = cv.flip(img, - 1)

cropped = img[200:400, 300:400]








cv.imshow('image', img)
cv.imshow('Translated', translated)
cv.imshow('Rotated', rotated) 
cv.imshow('Resized', resized)
cv.imshow('Flip', flip)
cv.imshow('Cropped', cropped)










# Wait for any key to close the window
cv.waitKey(0)




