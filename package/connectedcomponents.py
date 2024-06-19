import cv2 as cv
import numpy as np
def connected_detect(img_org):
    img = cv.cvtColor(img_org, cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(img,170,255,cv.THRESH_BINARY_INV)
    #cv.imwrite("t1.png", thresh)
    num, labels = cv.connectedComponents(thresh, 4)
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num):
        mask = labels == i
        output[:, :, 0][mask]= np.random.randint(0, 255)
        output[:, :, 1][mask]= np.random.randint(0, 255)
        output[:, :, 2][mask]= np.random.randint(0, 255)
    #cv.imwrite("t2.png", output)       
    return labels