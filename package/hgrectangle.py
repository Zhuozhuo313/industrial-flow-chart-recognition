import cv2
import numpy as np

def rectangle_detect(img):
    recList = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,230,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and area >= 10 and area <= 200000:
            w = abs(approx[2][0][0] - approx[0][0][0])
            h = abs(approx[2][0][1] - approx[0][0][1])
            area = cv2.contourArea(approx)
            if abs(area - w*h) <= 2*w + 2 and abs(area - w*h) <= 2*h + 2:
                recList.append(approx)               
    return recList            
    
if __name__ == '__main__':
    test_img = cv2.imread('cp.png')
    recList = rectangle_detect(test_img)
    for cnt in recList:
        test_img = cv2.drawContours(test_img, [cnt], -1, (0,255,0), 3)
    cv2.imwrite('rc.png', test_img)


