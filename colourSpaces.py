import cv2 as cv

import matplotlib.pyplot as plt


img = cv.imread('images/giraffe-zebra.jpg')

cv.imshow('giraffe-zebra', img)

plt.imshow(img)
plt.show()  

#BGR to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
cv.imshow('gray', gray)

#BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('hsv', hsv)

#BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('lab', lab)

#BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('rgb', rgb)

plt.imshow(rgb)
plt.show() 

cv.waitKey(0)