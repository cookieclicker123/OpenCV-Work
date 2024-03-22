import cv2 as cv
import numpy as np


img = cv.imread('images/baseball.jpg')

cv.imshow('baseball', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
cv.imshow('gray', gray)  


blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('blur', blur)

canny = cv.Canny(blur, 120, 175)
cv.imshow('canny edges', canny)

#ret , thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('thresh', thresh)  
contours , hierarchies = cv.findContours(canny , cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)  

cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)

print(f'{len(contours)} contour(s) found!')  

cv.waitKey(0)