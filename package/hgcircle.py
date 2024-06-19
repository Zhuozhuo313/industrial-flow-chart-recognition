import cv2
import numpy as np

def circle_detect(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 输出图像大小，方便根据图像大小调节minRadius和maxRadius
    # print(image.shape)

    # 霍夫变换圆检测
    circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1, 20, param1=300, param2=0.975, minRadius=10, maxRadius=80)
    # 图像平滑处理(高斯模糊)
    img=cv2.GaussianBlur(gray,(3,3),0)
    # 平滑处理后再进行一次霍夫变换圆检测
    circles2 = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1, 20, param1=300, param2=0.975, minRadius=10, maxRadius=80)
    if circles2 is None:
        if circles1 is None:
            circles = None
        else:
            circles = circles1
    elif circles1 is None:
        circles = circles2
    else:
        circles = np.concatenate((circles1,circles2), 1)
    return circles

if __name__ == '__main__':
    #img = cv2.imread('C:/Users/86521/Desktop/Eximg/test_280.png')
    img = cv2.imread('rc.png')
    circles = circle_detect(img)
    for circle in circles[0]:
        # 圆的基本信息
        print(circle[2])
        # 坐标行列－圆心坐标
        x = int(circle[0])
        y = int(circle[1])
        # 半径
        r = int(circle[2])
        # 在原图用指定颜色标记出圆的边界
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        # 画出圆的圆心
        cv2.circle(img, (x, y),1, (255, 0, 0), -1)
    cv2.imwrite("cc.png", img)