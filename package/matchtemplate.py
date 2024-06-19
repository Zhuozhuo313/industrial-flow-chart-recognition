import cv2 as cv
import numpy as np

#使用matchTemplate函数从图像中找到某个元素出现的所有位置
#参数：file      源图像文件完整路径（含文件名）                  例如：file = 'C:/Users/86521/Desktop/Eximg/test_40.png'
#参数：template  需要查找的模板元素的图像文件完整路径（含文件名）  例如：template = 'C:/Users/86521/arrow_tp.png'
#返回值：loc（一个元组，包含模板元素出现位置的左上角的所有坐标，包含两个数组，分别表示x坐标和y坐标），w，h（分别表示模板元素的宽和高）
def template_match(file,template,threshold):
    img_rgb = file
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    tp = cv.imread(template,0)
    w, h = tp.shape[::-1]

    res = cv.matchTemplate(img_gray,tp,cv.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)
    #print("w:%s h:%s" % (w,h))
    return loc, w, h

if __name__ == '__main__':
    loc1, w1, h1 = template_match('C:/Users/86521/Desktop/Eximg/test_280.png','C:/Users/86521/crossroad2_tp.png', 0.77)
    img_rgb1 = cv.imread('C:/Users/86521/Desktop/Eximg/test_280.png')
    for pt in zip(*loc1[::-1]):
        cv.rectangle(img_rgb1, pt, (pt[0] + w1, pt[1] + h1), (0,0,255), 2)
    cv.imwrite('res.png',img_rgb1)