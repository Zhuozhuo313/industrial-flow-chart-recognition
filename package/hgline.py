import cv2 as cv
import numpy as np
import copy

def line_detect(img_org):
    img_o2 = copy.deepcopy(img_org)
    img = cv.cvtColor(img_org, cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(img,210,255,cv.THRESH_BINARY_INV)
    #cv.imwrite("thr.png", thresh)
    #lines = cv.HoughLinesP(img, 0.1, np.pi/1800, 30, minLineLength = 20, maxLineGap = 6.5)
    solid_lines = cv.HoughLinesP(thresh, 0.1, np.pi/1800, 6, minLineLength = 30, maxLineGap = 1)
    if not solid_lines is None:
        for s_line in solid_lines: 
            x1, y1, x2, y2 = s_line[0]
            cv.line(img_o2, (x1, y1), (x2, y2), (255, 255, 255), 3)

    img_2 = cv.cvtColor(img_o2, cv.COLOR_BGR2GRAY)
    ret_2,thresh_2 = cv.threshold(img_2,210,255,cv.THRESH_BINARY_INV)
    dotted_lines = cv.HoughLinesP(thresh_2, 0.1, np.pi/1800, 6, minLineLength = 10, maxLineGap = 8)
    return solid_lines, dotted_lines

def line_detect_short(img_org):
    img = cv.cvtColor(img_org, cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(img,170,255,cv.THRESH_BINARY_INV)
    lines = cv.HoughLinesP(thresh, 0.1, np.pi/1800, 10, minLineLength = 10, maxLineGap = 3)
    return lines

if __name__ == '__main__':
    image_path="C:/Users/86521/Desktop/ImgInfExtraction/input/test_280.png"
    img_org = cv.imread(image_path)
    s, d = line_detect(img_org)
    for s_line in s: 
        x1, y1, x2, y2 = s_line[0]
        cv.line(img_org, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for d_line in d: 
        x1, y1, x2, y2 = d_line[0]
        cv.line(img_org, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imwrite('line_dt_test.png',img_org) 
