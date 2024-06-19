import time
from package.progressbar import *
start_time = time.perf_counter()

import cv2 as cv
import copy
import numpy as np
import argparse

from package.pointclass import *
from package.algorithmblockclass import *
from package.matchtemplate import template_match
from package.hgrectangle import rectangle_detect
from package.hgcircle import circle_detect
from package.wordocr import *
from package.hgline import *
from package.connectedcomponents import connected_detect
from package.operatorkeyword import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='test_280.png', required=False, help='The name of the input file.')
parser.add_argument('-o', '--output', type=str, default='test_280_output.png', required=False, help='The name of the output file.')
args = parser.parse_args()
file = './input/' + args.input
img = cv.imread(file)

# 裁剪掉多余的部分
crop = img [130:1895,249:3064]
crop_org = copy.deepcopy(crop)
crop_org [:,323] = (255,255,255)
crop_org [:,324] = (255,255,255)
crop_org [:,2557] = (255,255,255)
crop_org [:,2558] = (255,255,255)

# 不同类型的点集
StartPointList = []
EndPointList = []
NodeList = []
OperatorList = []

# 检测将要添加到点集中的点是否存在，防止重复添加
def repeatability_detect(x, y, exist_list, range=3):
    exist_flag = False
    for existing in exist_list:
        if abs(x - existing.x) < range and abs(y - existing.y) < range:
            exist_flag = True
            break
    return exist_flag  

# 将不接触的线处的"桥"改为十字
loc1, w1, h1 = template_match(crop,'./template/crossroad_tp.png',0.77)
for pt in zip(*loc1[::-1]):
    cv.rectangle(crop, pt, (pt[0] + 5, pt[1] + 5), (255,255,255), -1)
    cv.rectangle(crop, (pt[0] + 9, pt[1]), (pt[0] + 15 , pt[1] + 5), (255,255,255), -1)
    color0 = crop[pt[1] + 3, pt[0] + 6]
    color0 = tuple ([int(x) for x in color0])
    cv.rectangle(crop, (pt[0] + 6, pt[1]), (pt[0] + 6 , pt[1] + 5), color0, -1)
    color1 = crop[pt[1] + 3, pt[0] + 7]
    color1 = tuple ([int(x) for x in color1])
    cv.rectangle(crop, (pt[0] + 7, pt[1]), (pt[0] + 7 , pt[1] + 7), color1, -1)
    color2 = crop[pt[1] + 3, pt[0] + 8]
    color2 = tuple ([int(x) for x in color2])
    cv.rectangle(crop, (pt[0] + 8, pt[1]), (pt[0] + 8 , pt[1] + 7), color2, -1)
    color3 = crop[pt[1] + 6, pt[0]]
    color3 = tuple ([int(x) for x in color3])
    cv.rectangle(crop, (pt[0], pt[1] + 6), (pt[0] + 15 , pt[1] + 6), color3, -1)
    color4 = crop[pt[1] + 7, pt[0]]
    color4 = tuple ([int(x) for x in color4])
    cv.rectangle(crop, (pt[0], pt[1] + 7), (pt[0] + 15 , pt[1] + 7), color4, -1)
loc2, w2, h2 = template_match(crop,'./template/crossroad2_tp.png',0.77)
for pt in zip(*loc2[::-1]):
    cv.rectangle(crop, pt, (pt[0] + 5, pt[1] + 5), (255,255,255), -1)
    cv.rectangle(crop, (pt[0] + 9, pt[1]), (pt[0] + 15 , pt[1] + 5), (255,255,255), -1)
    color0 = crop[pt[1] + 3, pt[0] + 6]
    color0 = tuple ([int(x) for x in color0])
    cv.rectangle(crop, (pt[0] + 6, pt[1]), (pt[0] + 6 , pt[1] + 5), color0, -1)
    color1 = crop[pt[1] + 3, pt[0] + 7]
    color1 = tuple ([int(x) for x in color1])
    cv.rectangle(crop, (pt[0] + 7, pt[1]), (pt[0] + 7 , pt[1] + 7), color1, -1)
    color2 = crop[pt[1] + 3, pt[0] + 8]
    color2 = tuple ([int(x) for x in color2])
    cv.rectangle(crop, (pt[0] + 8, pt[1]), (pt[0] + 8 , pt[1] + 7), color2, -1)
    color3 = crop[pt[1] + 6, pt[0]]
    color3 = tuple ([int(x) for x in color3])
    cv.rectangle(crop, (pt[0], pt[1] + 6), (pt[0] + 15 , pt[1] + 6), color3, -1)
    color4 = crop[pt[1] + 7, pt[0]]
    color4 = tuple ([int(x) for x in color4])
    cv.rectangle(crop, (pt[0], pt[1] + 7), (pt[0] + 15 , pt[1] + 7), color4, -1)

# 识别并消去"带反转符的箭头"
loc3i, w3i, h3i = template_match(crop,'./template/arrowinv_tp.png',0.9)
for pt in zip(*loc3i[::-1]):
    cv.rectangle(crop, pt, (pt[0] + w3i - 3, pt[1] + h3i - 1), (255,255,255), -1)
    color1 = crop[pt[1] + 2, pt[0] + w3i -2]
    color1 = tuple ([int(x) for x in color1])
    cv.rectangle(crop, (pt[0] + w3i - 2, pt[1]), (pt[0] + w3i - 2, pt[1] + h3i - 1), color1, -1)
    color2 = crop[pt[1] + 2, pt[0] + w3i -1]
    color2 = tuple ([int(x) for x in color2])
    cv.rectangle(crop, (pt[0] + w3i - 1, pt[1]), (pt[0] + w3i - 1, pt[1] + h3i - 1), color2, -1)
    x = pt[0]
    y = pt[1] + (h3i-1) // 2
    if not repeatability_detect(x, y, EndPointList):        
        end_i = SorEPoint(x , y, "E")
        end_i.label_inv()
        EndPointList.append(end_i)

# 识别并消去"箭头"    
loc3, w3, h3 = template_match(crop,'./template/arrow_tp.png',0.77)
for pt in zip(*loc3[::-1]):
    cv.rectangle(crop, pt, (pt[0] + w3 - 2, pt[1] + h3 - 1), (255,255,255), -1)
    h3_2 = int(h3/2)
    color = crop[pt[1] + h3_2 -3, pt[0] + w3 -1]
    color = tuple ([int(x) for x in color])
    cv.rectangle(crop, (pt[0] + w3 - 1, pt[1] + h3_2 - 1), (pt[0] + w3 - 1, pt[1] + h3_2), color, -1)
    x = pt[0]
    y = pt[1] + (h3-1) // 2
    if not repeatability_detect(x, y, EndPointList):
        end_i = SorEPoint(x , y, "E")
        EndPointList.append(end_i)

# 识别并消去"交叉点"
loc4, w4, h4 = template_match(crop,'./template/node_tp.png',0.77)
for pt in zip(*loc4[::-1]):
    cv.rectangle(crop, pt, (pt[0] + w4 - 1, pt[1] + h4 - 1), (255,255,255), -1)
    x = pt[0] + (w4-1) // 2
    y = pt[1] + (h4-1) // 2
    if not repeatability_detect(x, y, NodeList):
        node_i = Node(x, y)
        NodeList.append(node_i)

# 消去右下角的黑色三角
loc5, w5, h5 = template_match(crop,'./template/triangle_tp.png',0.9)
for pt in zip(*loc5[::-1]):
    cv.rectangle(crop, pt, (pt[0] + w5 - 3, pt[1] + h5 - 3), (255,255,255), -1) 

#识别4圆角矩形算法块
loc10, w10, h10 = template_match(crop,'./template/rrc1_tp.png',0.8)
for pt in zip(*loc10[::-1]):
    cv.rectangle(crop, pt, (pt[0] + w10 - 1, pt[1] + h10 - 1), (255,255,255), -1) 
loc11, w11, h11 = template_match(crop,'./template/rrc2_tp.png',0.8)
for pt in zip(*loc11[::-1]):
    cv.rectangle(crop, pt, (pt[0] + w11 - 1, pt[1] + h11 - 1), (255,255,255), -1)
loc12, w12, h12 = template_match(crop,'./template/rrc3_tp.png',0.8)
for pt in zip(*loc12[::-1]):
    cv.rectangle(crop, pt, (pt[0] + w12 - 1, pt[1] + h12 - 1), (255,255,255), -1) 
loc13, w13, h13 = template_match(crop,'./template/rrc4_tp.png',0.75)
for pt in zip(*loc13[::-1]):
    cv.rectangle(crop, pt, (pt[0] + w13 - 1, pt[1] + h13 - 1), (255,255,255), -1)

RoundedRCList = []
for pt1 in zip(*loc10[::-1]):
    list1 = []
    for pt2 in zip(*loc11[::-1]):
        if pt2[0] - pt1[0] > 0 and abs(pt2[1] -pt1[1]) <= 2 :
            list1.append(pt2[0] + 19)
    if len(list1):
        cv.rectangle(crop, (pt1[0] - 1, pt1[1] - 1) , (min(list1) + 1, pt1[1] + 2), (255,255,255), -1)
        list2 = []
        for pt3 in zip(*loc12[::-1]):
            if pt3[1] - pt1[1] > 0 and abs(pt3[0] -pt1[0]) <= 2 :
                list2.append(pt3[1] + 19)    
        if len(list2):
            cv.rectangle(crop, (pt1[0] - 1, pt1[1] - 1) , (pt1[0] + 2, min(list2) + 1), (255,255,255), -1)
            cv.rectangle(crop, (min(list1) - 2, pt1[1] - 1) , (min(list1) + 1, min(list2) + 1), (255,255,255), -1)
            cv.rectangle(crop, (pt1[0] - 1, min(list2) - 2) , (min(list1) + 1, min(list2) + 1), (255,255,255), -1)
            rrc_i = RRC_Algorithm_Block(pt1[0] - 1, pt1[1] - 1 , min(list1) + 1, min(list2) + 1) #(左上角x,左上角y，右下角x，右下角y)
            exist_flag = False
            for existing in RoundedRCList:
                if abs(rrc_i.x1 - existing.x1) < 3 and abs(rrc_i.y1 - existing.y1) < 3:
                    exist_flag = True
                    break
            if not exist_flag:
                RoundedRCList.append(rrc_i)

#for rrc in RoundedRCList:
#    cv.rectangle(crop_org, (rrc[0], rrc[1]) , (rrc[2], rrc[3]), (0,255,0), 3)

RoundedTwoRCList = []

# 识别2圆角矩形算法块
loc15, w15, h15 = template_match(crop,'./template/rrc2_2_tp.png',0.8)
for pt in zip(*loc15[::-1]):
    cv.rectangle(crop, pt, (pt[0] + w15 - 1, pt[1] + h15 - 1), (255,255,255), -1)
    #cv.rectangle(crop_org, pt, (pt[0] + w15 - 1, pt[1] + h15 - 1), (0,255,255), 3)

loc14, w14, h14 = template_match(crop,'./template/rrc2_1_tp.png',0.8)
for pt in zip(*loc14[::-1]):
    cv.rectangle(crop, pt, (pt[0] + w14 - 1, pt[1] + h14 - 1), (255,255,255), -1)
    #cv.rectangle(crop_org, pt, (pt[0] + w14 - 1, pt[1] + h14 - 1), (0,255,255), 1)
    list1 = []
    for pt2 in zip(*loc15[::-1]):
        if pt2[0] - pt[0] > 0 and abs(pt2[1] -pt[1]) <= 3 :
            list1.append(pt2[0] + 25)
    if len(list1):
        cv.rectangle(crop, (pt[0] - 1, pt[1] - 1) , (min(list1) + 1, pt[1] + 2), (255,255,255), -1)
        cv.rectangle(crop, (pt[0] - 1, pt[1] + 21) , (pt[0] + 3, pt[1] + 41), (255,255,255), -1)
        cv.rectangle(crop, (min(list1) - 3, pt[1] + 21) , (min(list1) + 1, pt[1] + 41), (255,255,255), -1)
        rtrc_i = r2rc_Algorithm_Block(pt[0] - 1, pt[1] - 1 , min(list1) + 1, pt[1] + 41)
        exist_flag = False
        for existing in RoundedTwoRCList:
            if abs(rtrc_i.x1 - existing.x1) < 3 and abs(rtrc_i.y1 - existing.y1) < 3:
                exist_flag = True
                break
        if not exist_flag:
            RoundedTwoRCList.append(rtrc_i)
#cv.imwrite("test.png", crop)

# 识别矩形
RectangleList = rectangle_detect(crop)

# 以下的数组用于识别算法块
Rectangle_withword_List = []
Rectangle_inprocess_List = []
Rectangle_notrepeating_List = []

# 以下的数组用于识别运算符
Rectangle_op_List = []
for rect in RectangleList:
    # 防止矩形重复
    rc_repeat_flag = False
    for rtwi in Rectangle_op_List:
        if (abs(rect[0][0][0] - rtwi[0][0][0][0]) < 5 and abs(rect[2][0][0] - rtwi[0][2][0][0]) < 5 and
            abs(rect[0][0][1] - rtwi[0][0][0][1]) < 5 and abs(rect[2][0][1] - rtwi[0][2][0][1]) < 5):
            rc_repeat_flag = True
            break
    if not rc_repeat_flag:
        Rectangle_op_List.append([rect,[]])

progress_bar(5,start_time)
# 将文字绑定到算法块
ocr_result0 = word_ocr_nocleaner(crop)
for idx in range(len(ocr_result0)):
    res = ocr_result0[idx]
    for line in res:
        # print(line)
        po1 = int(line[0][0][0])
        po2 = int(line[0][0][1])
        po3 = int(line[0][2][0])
        po4 = int(line[0][2][1])
        word_x = (po1 + po3) / 2
        word_y = (po2 + po4) / 2
        for rt in RectangleList:
            if word_x > rt[0][0][0] and word_x < rt[2][0][0] and word_y > rt[0][0][1] and word_y < rt[2][0][1]:
                Rectangle_withword_List.append((line,rt))
                break
        for rrc in RoundedRCList:
            if word_x > rrc.x1 and word_x < rrc.x2 and word_y > rrc.y1 and word_y < rrc.y2:
                rrc.any_word.append(line)
                break
        for r2rc in RoundedTwoRCList:
            if word_x > r2rc.x1 and word_x < r2rc.x2 and word_y > r2rc.y1 and word_y < r2rc.y2:
                r2rc.title = line
                break
        for rtwi in Rectangle_op_List:
            if word_x > rtwi[0][0][0][0] and word_x < rtwi[0][2][0][0] and word_y > rtwi[0][0][0][1] and word_y < rtwi[0][2][0][1]:
                rtwi[1].append(line)
                break
progress_bar(25,start_time)            
# 4圆角功能块内部识别
for rrc in RoundedRCList:
    w = rrc.x2 - rrc.x1
    h = rrc.y2 - rrc.y1
    left_notsorted_list = []
    right_notsorted_list = []
    middle_notsorted_list = []

    for line in rrc.any_word:
        po1 = int(line[0][0][0])
        po2 = int(line[0][0][1])
        po3 = int(line[0][2][0])
        po4 = int(line[0][2][1])
        word_x = (po1 + po3) / 2
        word_y = (po2 + po4) / 2
        if word_x < rrc.x1 + w/3:
            left_notsorted_list.append(line)
        elif word_x > rrc.x1 + (2*w)/3:
            right_notsorted_list.append(line)
        else:
            middle_notsorted_list.append(line)
    middle_notsorted_list.sort(key = lambda l:l[0][0][1])
    rrc.title = middle_notsorted_list[0]
    if len(left_notsorted_list) and len(right_notsorted_list):
        left_notsorted_list.sort(key = lambda l:l[0][0][1])
        right_notsorted_list.sort(key = lambda l:l[0][0][1])
        if (middle_notsorted_list[1][0][2][0] - middle_notsorted_list[1][0][0][0]) < (middle_notsorted_list[1][0][2][1] - middle_notsorted_list[1][0][0][1]):
            rrc.type = 3
            rrc_pic = crop[rrc.y1:rrc.y2,rrc.x1:rrc.x2]
            h, w = rrc_pic.shape[:2]
            padding = (h - w) // 2
            center = (h // 2, h // 2)
            rrc_pic_padded = np.zeros(shape=(h, h, 3), dtype=np.uint8)
            rrc_pic_padded[:, padding:padding+w, :] = rrc_pic
            M = cv.getRotationMatrix2D(center, -90, 1)
            rotated_padded = cv.warpAffine(rrc_pic_padded, M, (h, h))
            rrc_rt = rotated_padded[padding:padding+w, :, :]
            rt_ocr = word_ocr_nocleaner(rrc_rt)
            for idx in range(len(rt_ocr)):
                res = rt_ocr[idx]
                for line in res:
                    #print(line)
                    po1 = int(line[0][0][0])
                    po2 = int(line[0][0][1])
                    po3 = int(line[0][2][0])
                    po4 = int(line[0][2][1])
                    #txts = line[1][0]
                    word_x = (po1 + po3) / 2
                    word_y = (po2 + po4) / 2
                    if word_x > h/3 and word_x < (2*h/3) and word_y > w/3 and word_y < (2*w/3):
                        line[0][0][0] = rrc.x1 + po2
                        line[0][0][1] = rrc.y1 + h - po3
                        line[0][2][0] = rrc.x1 + po4
                        line[0][2][1] = rrc.y1 + h - po1
                        rrc.middle = line
                        cv.rectangle(rrc_pic, (po2, h - po3), (po4, h - po1), (255, 255, 255),-1)
                        break
            rrc_ocr = word_ocr_nocleaner(rrc_pic)
            for idx in range(len(rrc_ocr)):
                res = rrc_ocr[idx]
                for line in res:
                    #print(line)
                    po1 = int(line[0][0][0])
                    po2 = int(line[0][0][1])
                    po3 = int(line[0][2][0])
                    po4 = int(line[0][2][1])
                    #txts = line[1][0]
                    word_x = (po1 + po3) / 2
                    word_y = (po2 + po4) / 2
                    line[0][0][0] += rrc.x1
                    line[0][0][1] += rrc.y1
                    line[0][2][0] += rrc.x1
                    line[0][2][1] += rrc.y1
                    if word_x < w/3:
                        rrc.left_list.append(line)
                    elif word_x > (2*w)/3:
                        rrc.right_list.append(line)              
                    else:
                        rrc.bottom_list.append(line)      
            rrc.left_list.sort(key = lambda l:l[0][0][1])
            rrc.right_list.sort(key = lambda l:l[0][0][1])
            rrc.bottom_list.sort(key = lambda l:l[0][0][1])
            rrc.bottom_list.pop(0)

        else:
            rrc.type = 2
            rrc.middle = middle_notsorted_list[1]
            for i in range(2,len(middle_notsorted_list)):
                rrc.bottom_list.append(middle_notsorted_list[i])
            rrc.right_list = right_notsorted_list
            rrc.left_list = left_notsorted_list
    else:
        rrc.type = 1
        for i in range(1,len(middle_notsorted_list)):
            rrc.word_list.append(middle_notsorted_list[i])
progress_bar(50,start_time)
# 识别矩形算法块
# 防止重复
for rtww in Rectangle_withword_List:
    min_flag = True
    if len(Rectangle_inprocess_List):
        for existing in Rectangle_inprocess_List:
            if not existing[1]:
                continue
            elif abs(rtww[1][0][0][0] - existing[0][1][0][0][0]) < 3 and abs(rtww[1][0][0][1] - existing[0][1][0][0][1]) < 3:
                if rtww[1][2][0][0] - existing[0][1][2][0][0] > -3 or rtww[1][2][0][1] - existing[0][1][2][0][1] > -3:
                    min_flag = False
                else:
                    Rectangle_inprocess_List.append([rtww,True])
                    existing[1] = False
                break
        if min_flag:
            Rectangle_inprocess_List.append([rtww,True])
    else:
        Rectangle_inprocess_List.append([rtww,True])
#i = 0
for rtip in Rectangle_inprocess_List:
    if rtip[1]:
        Rectangle_notrepeating_List.append([rtip[0],True])
        #cv.putText(crop_org, str(i), (rtip[0][1][0][0][0],rtip[0][1][0][0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        #cv.rectangle(crop_org, (rtip[0][1][0][0][0],rtip[0][1][0][0][1]) , (rtip[0][1][2][0][0],rtip[0][1][2][0][1]), (0,0,255), 1)
        #i+=1
    #cv.rectangle(crop_org, (rtww[1][0][0][0],rtww[1][0][0][1]) , (rtww[1][2][0][0],rtww[1][2][0][1]), (0,0,255), 3)

# 为2圆角算法块绑定内容
for r2rc in RoundedTwoRCList:
    for rt in Rectangle_notrepeating_List:
        if abs(r2rc.y2 - rt[0][1][0][0][1]) < 10 and abs(r2rc.x1 - rt[0][1][0][0][0]) < 10 and abs(r2rc.x2 - rt[0][1][2][0][0]) < 10:
            r2rc.word_list.append(rt[0])
            rt[1] = False
            break
    current_rt = r2rc.word_list[0]
    search_flag = True
    while search_flag:
        search_flag = False
        for rt in Rectangle_notrepeating_List:
            if not rt[1]:
                continue
            elif abs(current_rt[1][2][0][1] - rt[0][1][0][0][1]) < 6 and abs(current_rt[1][0][0][0] - rt[0][1][0][0][0]) < 6 and abs(current_rt[1][2][0][0] - rt[0][1][2][0][0]) < 6:
                r2rc.word_list.append(rt[0])
                current_rt = rt[0]
                rt[1] = False
                search_flag = True
                break

# 定义字符串模糊函数
def normalize(s):
    s = s.replace("O", "0")
    s = s.replace("I", "l")
    s = s.upper()
    return s

# 运算符文字识别
for i in range(len(Rectangle_op_List)-1,-1,-1):
    rtwi = Rectangle_op_List[i]
    for key in operator_keyword_dict:
        list_not_match_flag = False
        kw_list = operator_keyword_dict[key]
        for kw in kw_list:
            kw_match_flag = False
            for line in rtwi[1]:
                po1 = int(line[0][0][0])
                po2 = int(line[0][0][1])
                po3 = int(line[0][2][0])
                po4 = int(line[0][2][1])
                word_x = (po1 + po3) / 2
                word_y = (po2 + po4) / 2
                content = line[1][0]
                content_split = content.split()
                for ct in content_split:
                    #print(ct)
                    if normalize(ct) == normalize(kw) or normalize(''.join(reversed(ct))) == normalize(kw):
                        kw_match_flag = True
                        break
            if not kw_match_flag:
                list_not_match_flag = True
                break
        if not list_not_match_flag:
            x1 = min(rtwi[0][0][0][0],rtwi[0][1][0][0],rtwi[0][2][0][0],rtwi[0][3][0][0]) - 2
            x2 = max(rtwi[0][0][0][0],rtwi[0][1][0][0],rtwi[0][2][0][0],rtwi[0][3][0][0]) + 2
            y1 = min(rtwi[0][0][0][1],rtwi[0][1][0][1],rtwi[0][2][0][1],rtwi[0][3][0][1]) - 2
            y2 = max(rtwi[0][0][0][1],rtwi[0][1][0][1],rtwi[0][2][0][1],rtwi[0][3][0][1]) + 2
            #cv.rectangle(crop_org, (x1,y1) , (x2,y2), (0,0,255), 3)
            op_i = OpPoint(int((x1+x2)/2) , int((y1+y2)/2), key)
            op_i.rect_inf(int(x1), int(y1), int(x2), int(y2))
            if not repeatability_detect(op_i.x, op_i.y, OperatorList, 20):
                OperatorList.append(op_i)
                Rectangle_op_List.pop(i)
            break
#cv.imwrite("tr.png",crop_org)
# 运算符模板匹配
crop_expand = cv.copyMakeBorder(crop,53,53,53,53, cv.BORDER_CONSTANT,value=[255,255,255])
for priority in range(3):
    threshold = 0.99
    while threshold >= 0.7:
        for i in range(len(Rectangle_op_List)-1,-1,-1):
            rtm = Rectangle_op_List[i]
            tm = rtm[0]
            tm_pic = crop_expand[tm[0][0][1]+3:tm[2][0][1]+103,tm[0][0][0]+3:tm[2][0][0]+103]
            for key in operator_template_dict:
                if operator_template_dict[key][1] == priority:
                    key_match_flag = False
                    tp = operator_template_dict[key][0]
                    tp_address = './template/' + tp + '_tp.png'
                    loctp, wtp, htp = template_match(tm_pic,tp_address,threshold)
                    for pt in zip(*loctp[::-1]):
                        x1 = min(tm[0][0][0],tm[1][0][0],tm[2][0][0],tm[3][0][0]) - 2
                        x2 = max(tm[0][0][0],tm[1][0][0],tm[2][0][0],tm[3][0][0]) + 2
                        y1 = min(tm[0][0][1],tm[1][0][1],tm[2][0][1],tm[3][0][1]) - 2
                        y2 = max(tm[0][0][1],tm[1][0][1],tm[2][0][1],tm[3][0][1]) + 2
                        optype = key.strip('*')
                        op_i = OpPoint(int((x1+x2)/2) , int((y1+y2)/2), optype)
                        op_i.rect_inf(int(x1), int(y1), int(x2), int(y2))  
                        if not repeatability_detect(op_i.x, op_i.y, OperatorList, 20):
                            OperatorList.append(op_i)
                            Rectangle_op_List.pop(i)
                            key_match_flag = True
                            break
                    if key_match_flag:
                        break
        threshold -= 0.01

# 将SVR/SRV相邻的两块整合成一块
for i in range(len(OperatorList)-1,-1,-1):
    op_i = OperatorList[i]
    if op_i.optype == 'SVR' or op_i.optype == 'SRV':
        for j in range(len(OperatorList)-1,-1,-1):
            if j != i:
                op_j = OperatorList[j]
                if op_j.optype == op_i.optype:
                    if (abs(op_i.rectangle[0] - op_j.rectangle[0]) < 5 and abs(op_i.rectangle[2] - op_j.rectangle[2]) < 5 and
                        (abs(op_i.rectangle[3] - op_j.rectangle[1]) < 7 or abs(op_i.rectangle[1] - op_j.rectangle[3]) < 7)):
                        OperatorList.pop(i)
                        OperatorList.pop(j)
                        x1 = op_i.rectangle[0]
                        x2 = op_i.rectangle[2]
                        y1 = min(op_i.rectangle[1],op_j.rectangle[1])
                        y2 = max(op_i.rectangle[3],op_j.rectangle[3])
                        op_ij = OpPoint(int((x1+x2)/2) , int((y1+y2)/2), op_i.optype)
                        op_ij.rect_inf(int(x1), int(y1), int(x2), int(y2))
                        if not repeatability_detect(op_ij.x, op_ij.y, OperatorList, 20):
                            OperatorList.append(op_ij)
                          
# 为运算器配对终点并添加起点
for op_i in OperatorList:
    for end_i in EndPointList:
        dis_x = op_i.rectangle[0] - end_i.x
        if dis_x < 40 and dis_x > -10 and end_i.y > op_i.rectangle[1] and end_i.y < op_i.rectangle[3]:
            op_i.link_SorE(end_i.x, end_i.y, "E")

for i in range(len(OperatorList)-1,-1,-1):
    if (OperatorList[i].num_End == 0 and (abs((OperatorList[i].rectangle[2] - OperatorList[i].rectangle[0])/(OperatorList[i].rectangle[3] - OperatorList[i].rectangle[1])) > 2 or
        OperatorList[i].rectangle[2] - OperatorList[i].rectangle[0] < 30 or OperatorList[i].rectangle[3] - OperatorList[i].rectangle[1] < 30 or
        abs((OperatorList[i].rectangle[3] - OperatorList[i].rectangle[1])/(OperatorList[i].rectangle[2] - OperatorList[i].rectangle[0])) > 2.5)):
        OperatorList.pop(i)

for op_i in OperatorList:
    if not repeatability_detect(op_i.rectangle[2], op_i.y, StartPointList):
        if op_i.optype in start_special_distance:
            start_i = SorEPoint(op_i.rectangle[2] , op_i.rectangle[1] + start_special_distance[op_i.optype], "S")
        else:
            start_i = SorEPoint(op_i.rectangle[2] , op_i.y, "S")
        StartPointList.append(start_i)
        op_i.link_SorE(start_i.x, start_i.y, "S")

# 在原图删除识别到的运算符
for op_i in OperatorList:
    cv.rectangle(crop, (op_i.rectangle[0],op_i.rectangle[1]) , (op_i.rectangle[2],op_i.rectangle[3]), (255,255,255), -1)
# cv.imwrite("optest.png",crop)
# 判断运算符是否有黑色三角
for op_i in OperatorList:
    if len(op_i.rectangle):
        for pt in zip(*loc5[::-1]):
            if (pt[0] + int((w5-3)/2) < op_i.rectangle[2] and pt[0] + int((w5-3)/2) > op_i.rectangle[0] and
                pt[1] + int((h5-3)/2) < op_i.rectangle[3] and pt[1] + int((h5-3)/2) > op_i.rectangle[1]):
                op_i.if_triangle = True
                break

# 将小矩形块组合成矩形功能块
RectangleABList = []
digits = set("0123456789")
for rt in Rectangle_notrepeating_List:
    if not rt[1] or len(rt[0][0][1][0]) < 2:
        continue
    elif rt[0][0][1][0][0] in digits and rt[0][0][1][0][1] in digits:
        RectangleABList.append(rectangle_Algorithm_Block(rt[0][1][0][0][0],rt[0][1][0][0][1],rt[0][1][2][0][0],rt[0][1][2][0][1],rt[0][0]))
        rt[1] = False
for rab in RectangleABList:
    for rt in Rectangle_notrepeating_List:
        if not rt[1]:
            continue
        elif abs(rab.y2 - rt[0][1][0][0][1]) < 10 and abs(rab.x1 - rt[0][1][0][0][0]) < 10 and rt[0][1][2][0][0] > rab.x1 and rt[0][1][2][0][0] < rab.x2:
            rab.left_list.append(rt[0])
            rt[1] = False
        elif abs(rab.y2 - rt[0][1][0][0][1]) < 10 and abs(rab.x2 - rt[0][1][2][0][0]) < 10 and rt[0][1][0][0][0] > rab.x1 and rt[0][1][0][0][0] < rab.x2:
            rab.right_list.append(rt[0])
            rt[1] = False
for rab in RectangleABList:
    if (not len(rab.left_list)) or (not len(rab.right_list)):
        continue
    current_rt_l = rab.left_list[0]
    search_flag_l = True
    while search_flag_l:
        search_flag_l = False
        for rt in Rectangle_notrepeating_List:
            if not rt[1]:
                continue
            elif abs(current_rt_l[1][2][0][1] - rt[0][1][0][0][1]) < 6 and abs(current_rt_l[1][0][0][0] - rt[0][1][0][0][0]) < 6 and abs(current_rt_l[1][2][0][0] - rt[0][1][2][0][0]) < 6:
                rab.left_list.append(rt[0])
                current_rt_l = rt[0]
                rt[1] = False
                search_flag_l = True
                break
    current_rt_r = rab.right_list[0]
    search_flag_r = True
    while search_flag_r:
        search_flag_r = False
        for rt in Rectangle_notrepeating_List:
            if not rt[1]:
                continue
            elif abs(current_rt_r[1][2][0][1] - rt[0][1][0][0][1]) < 6 and abs(current_rt_r[1][0][0][0] - rt[0][1][0][0][0]) < 6 and abs(current_rt_r[1][2][0][0] - rt[0][1][2][0][0]) < 6:
                rab.right_list.append(rt[0])
                current_rt_r = rt[0]
                rt[1] = False
                search_flag_r = True
                break

# 为识别到的功能块添加起点
for r2rc in RoundedTwoRCList:
    len_word = len(r2rc.word_list)
    y_bottom = int(r2rc.word_list[len_word-1][1][2][0][1])
    cv.rectangle(crop, (r2rc.x1,r2rc.y1) , (r2rc.x2,y_bottom+2), (255,255,255), -1)
    for w in r2rc.word_list:
        x = w[1][2][0][0] + 2
        y = int((w[1][0][0][1] + w[1][2][0][1])/2)
        start_i = SorEPoint(x, y, "S")
        StartPointList.append(start_i)
        r2rc.start_list.append((start_i,w))
for rab in RectangleABList:
    len_right = len(rab.right_list)
    y_bottom = int(rab.right_list[len_right-1][1][2][0][1])
    cv.rectangle(crop, (rab.x1-2,rab.y1-2) , (rab.x2+2,y_bottom+2), (255,255,255), -1)
    for r in rab.right_list:
        x = r[1][2][0][0] + 2
        y = int((r[1][0][0][1] + r[1][2][0][1])/2)
        start_i = SorEPoint(x, y, "S")
        StartPointList.append(start_i)
        rab.start_list.append((start_i,r))
for rrc in RoundedRCList:
    cv.rectangle(crop, (rrc.x1,rrc.y1) , (rrc.x2,rrc.y2), (255,255,255), -1)
    if rrc.type == 1:
        for w in rrc.word_list:
            start_flag = False
            nearby_area = crop[int(w[0][0][1]):int(w[0][2][1]), rrc.x2:rrc.x2+20]
            lines = line_detect_short(nearby_area)
            start_flag = False
            if not lines is None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if (y2-y1)/(x2-x1) < 0.3:
                        start_flag = True
                        break
            if start_flag:
                start_i = SorEPoint(rrc.x2,int((w[0][0][1]+w[0][2][1])/2), "S")
                StartPointList.append(start_i)
                rrc.start_list.append((start_i,w))
            for end_i in EndPointList:
                if abs(end_i.x - rrc.x1) < 20 and end_i.y > int(w[0][0][1]) and end_i.y < int(w[0][2][1]):
                    rrc.end_list.append((end_i,w))
    else:
        for r in rrc.right_list:
            start_flag = False
            nearby_area = crop[int(r[0][0][1]):int(r[0][2][1]), rrc.x2:rrc.x2+20]
            lines = line_detect_short(nearby_area)
            start_flag = False
            if not lines is None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if (y2-y1)/(x2-x1) < 0.3:
                        start_flag = True
                        break
            if start_flag:
                start_i = SorEPoint(rrc.x2,int((r[0][0][1]+r[0][2][1])/2), "S")
                StartPointList.append(start_i)
                rrc.start_list.append((start_i,r))
        for l in rrc.left_list:
            for end_i in EndPointList:
                if abs(end_i.x - rrc.x1) < 20 and end_i.y > int(l[0][0][1]) and end_i.y < int(l[0][2][1]):
                    rrc.end_list.append((end_i,l))

# 识别终点箭头
Endarrow_List = []
loc03, w03, h03 = template_match(crop,'./template/endarrow_tp.png',0.8)
for pt in zip(*loc03[::-1]):
    cv.rectangle(crop, pt, (pt[0] + w03 - 1, pt[1] + h03 - 1), (255,255,255), -1)
    x = pt[0] + (w03-1)//2
    y = pt[1] + (h03-1)//2
    x_end = pt[0]
    if not repeatability_detect(x, y, Endarrow_List): 
        ea_i = circle_Algorithm_Block(x, y, -1, 4)
        ea_i.rect_inf(pt[0],pt[1],pt[0] + w03 - 1, pt[1] + h03 - 1)
        Endarrow_List.append(ea_i)
        if not repeatability_detect(x_end, y, EndPointList): 
            end_i = SorEPoint(x_end, y, "E")
            ea_i.if_start = False
            ea_i.linked_point = end_i
            EndPointList.append(end_i)

# 识别操场形
Playground_List = []
loc04, w04, h04 = template_match(crop,'./template/playground_tp.png',0.8)
for pt in zip(*loc04[::-1]):
    x = pt[0] + (w04-1)//2
    y = pt[1] - 6
    x_start = pt[0] + w04 - 1
    x_end = pt[0]
    if not repeatability_detect(x, y, Playground_List): 
        pg_i = circle_Algorithm_Block(x, y, -1, 5)
        pg_i.rect_inf(pt[0],pt[1] + h04 - 36,pt[0] + w04 - 1, pt[1] + h04 - 1)
        Playground_List.append(pg_i)
        if x < 1407:
            pg_i.if_start = True
            if not repeatability_detect(x_start, y, StartPointList): 
                start_i = SorEPoint(x_start, y, "S")
                pg_i.linked_point = start_i
                StartPointList.append(start_i)
        else:
            pg_i.if_start = False
            for end_i in EndPointList:
                if abs(x_end - end_i.x) < 6 and abs(y - end_i.y) < 6 :
                    pg_i.linked_point = end_i
                    break

# 识别圆形
RecognizedCircleList = circle_detect(crop)
cc_exist = []
repeat_flag = False
SolidCircleList = []
if not RecognizedCircleList is None:
    for circle in RecognizedCircleList[0]:
        # 坐标行列－圆心坐标
        x = int(circle[0])
        y = int(circle[1])
        # 半径
        r = int(circle[2])
        # 消去圆形轮廓
        cv.circle(crop, (x, y), r, (255, 255, 255), 2)
        # 防止重复
        for (xe, ye) in cc_exist:
            if (x-xe)**2 + (y-ye)**2 < r**2:
                repeat_flag = True
        if not repeat_flag:
            sc_i = circle_Algorithm_Block(x, y, r, 2)
            SolidCircleList.append(sc_i)
            if x < 1407:  
                # 加入起点列表
                sc_i.if_start = True
                if not repeatability_detect(sc_i.x + sc_i.r, sc_i.y, StartPointList): 
                    start_i = SorEPoint(sc_i.x + sc_i.r, sc_i.y, "S")
                    StartPointList.append(start_i)
                    sc_i.linked_point = start_i
            else:
                sc_i.if_start = False
                for end_i in EndPointList:
                    if abs(sc_i.x - sc_i.r - end_i.x) < 6 and abs(sc_i.y - end_i.y) < 6 :
                        sc_i.linked_point = end_i
                        break
                # 加入终点列表
                # end_i = SorEPoint(x-r, y, "E")
                # EndPointList.append(end_i)
        cc_exist.append((x, y))
        repeat_flag = False

DashedCircleList = []
loc16, w16, h16 = template_match(crop,'./template/dashedcircle_tp.png',0.75)
for pt in zip(*loc16[::-1]):
    dc_i = circle_Algorithm_Block(pt[0] + 26, pt[1] - 6, 28, 1)
    DashedCircleList.append(dc_i)
    cv.circle(crop, (dc_i.x, dc_i.y), dc_i.r, (255, 255, 255), 3)
    cv.circle(crop_org, (dc_i.x, dc_i.y), dc_i.r, (0, 255, 255), 3)
    if dc_i.x < 1407:
        dc_i.if_start = True
        if not repeatability_detect(dc_i.x + dc_i.r, dc_i.y, StartPointList): 
            start_i = SorEPoint(dc_i.x + dc_i.r, dc_i.y, "S")
            StartPointList.append(start_i)
            dc_i.linked_point = start_i
    else:
        dc_i.if_start = False
        for end_i in EndPointList:
            if abs(dc_i.x - dc_i.r - end_i.x) < 6 and abs(dc_i.y - end_i.y) < 6 :
                dc_i.linked_point = end_i
                break
cv.imwrite('crop_org.png',crop_org)

# 识别六边形
Hexagon_List = []
loc05, w05, h05 = template_match(crop,'./template/hexagon_tp.png',0.7)
for pt in zip(*loc05[::-1]):
    hex_i = circle_Algorithm_Block(pt[0] + 29, pt[1] - 6, 32, 3)
    Hexagon_List.append(hex_i)
    if hex_i.x < 1407:
        hex_i.if_start = True
        if not repeatability_detect(hex_i.x + hex_i.r, hex_i.y, StartPointList): 
            start_i = SorEPoint(hex_i.x + hex_i.r, hex_i.y, "S")
            StartPointList.append(start_i)
            hex_i.linked_point = start_i
    else:
        hex_i.if_start = False
        for end_i in EndPointList:
            if abs(hex_i.x - hex_i.r - end_i.x) < 6 and abs(hex_i.y - end_i.y) < 6 :
                hex_i.linked_point = end_i
                break

# 合并圆形list
CircleList = DashedCircleList + SolidCircleList

# 裁剪多余的线防止影响连通性判断
crop [:,323] = (255,255,255)
crop [:,324] = (255,255,255)
crop [:,2557] = (255,255,255)
crop [:,2558] = (255,255,255)

# 图像连通性判断
crop_con = copy.deepcopy(crop)

# 恢复"交叉点"
for pt in zip(*loc4[::-1]):
    pt[0] + (w4-1) // 2, pt[1] + (h4-1) // 2
    cv.rectangle(crop_con, (pt[0] + (w4-1) // 2 - 1, pt[1]), (pt[0] + (w4-1) // 2 + 1 , pt[1] + h4 - 1), (0, 0, 0), -1)
    cv.rectangle(crop_con, (pt[0], pt[1] + (h4-1) // 2 - 1), (pt[0] + w4 - 1, pt[1] + (h4-1) // 2 + 1), (0, 0, 0), -1)

# 恢复"起点"
for start_i in StartPointList:
    cv.rectangle(crop_con, (start_i.x, start_i.y - 1), (start_i.x + 2, start_i.y + 1), (0, 0, 0), -1)

# 恢复"终点"
for pt in zip(*loc3i[::-1]):
    cv.rectangle(crop_con, (pt[0] - 1, pt[1] + (h3i-1) // 2), (pt[0], pt[1] + (h3i-1) // 2 + 1), (0, 0, 0), -1)
for pt in zip(*loc3[::-1]):
    cv.rectangle(crop_con, (pt[0] - 1, pt[1] + (h3-1) // 2), (pt[0], pt[1] + (h3-1) // 2 + 1), (0, 0, 0), -1)
# cv.imwrite("pic_con.png", crop_con)
crop_ver = copy.deepcopy(crop_con)
for pt in zip(*loc1[::-1]):
    cv.rectangle(crop_ver, (pt[0], pt[1] + 6), (pt[0] + 6 , pt[1] + 7), (255,255,255), -1)
    cv.rectangle(crop_ver, (pt[0] + 9, pt[1] + 6), (pt[0] + 15 , pt[1] + 7), (255,255,255), -1)
    cv.rectangle(crop_ver, (pt[0] + 7, pt[1] + 6), (pt[0] + 8 , pt[1] + 8), (0,0,0), -1)
for pt in zip(*loc2[::-1]):
    cv.rectangle(crop_ver, (pt[0], pt[1] + 6), (pt[0] + 6 , pt[1] + 7), (255,255,255), -1)
    cv.rectangle(crop_ver, (pt[0] + 9, pt[1] + 6), (pt[0] + 15 , pt[1] + 7), (255,255,255), -1)
    cv.rectangle(crop_ver, (pt[0] + 7, pt[1] + 6), (pt[0] + 8 , pt[1] + 8), (0,0,0), -1)
# cv.imwrite("pic_ver.png", crop_ver)
connected_V = connected_detect(crop_ver)

crop_hor = copy.deepcopy(crop_con)
for pt in zip(*loc1[::-1]):
    cv.rectangle(crop_hor, (pt[0] + 6, pt[1] - 1), (pt[0] + 8 , pt[1] + 5), (255,255,255), -1)
    cv.rectangle(crop_hor, (pt[0] + 6, pt[1] + 8), (pt[0] + 8 , pt[1] + 14), (255,255,255), -1)
for pt in zip(*loc2[::-1]):
    cv.rectangle(crop_hor, (pt[0] + 6, pt[1] - 1), (pt[0] + 8 , pt[1] + 5), (255,255,255), -1)
    cv.rectangle(crop_hor, (pt[0] + 6, pt[1] + 8), (pt[0] + 8 , pt[1] + 14), (255,255,255), -1)
# cv.imwrite("pic_hor.png", crop_hor)
connected_H = connected_detect(crop_hor)

# 根据连通性进行配对
for node_i in NodeList:
    for node_j in NodeList:
        if (node_i.x == node_j.x and node_i.y == node_j.y) or node_i.if_linked_with(node_j.x, node_j.y):
            continue
        elif connected_H[node_i.y, node_i.x] == connected_H[node_j.y, node_j.x] or connected_V[node_i.y, node_i.x] == connected_V[node_j.y, node_j.x]:
            node_i.link_other_Node(node_j.x, node_j.y)
            node_j.link_other_Node(node_i.x, node_i.y)

for start_i in StartPointList:
    for node_i in NodeList:
        if connected_H[start_i.y, start_i.x] == connected_H[node_i.y, node_i.x] or connected_V[start_i.y, start_i.x] == connected_V[node_i.y, node_i.x]:
            start_i.matchPoint(node_i.x, node_i.y)
            node_i.matchOnePoint(start_i.x,start_i.y,"S")
            break
    if start_i.xm == -1:
        for end_i in EndPointList:
            if connected_H[start_i.y, start_i.x] == connected_H[end_i.y, end_i.x] or connected_V[start_i.y, start_i.x] == connected_V[end_i.y, end_i.x]:
                start_i.matchPoint(end_i.x, end_i.y)
                end_i.matchPoint(start_i.x, start_i.y)
                break

for end_i in EndPointList:
    if end_i.xm != -1:
        continue
    for node_i in NodeList:
        if connected_H[end_i.y, end_i.x] == connected_H[node_i.y, node_i.x] or connected_V[end_i.y, end_i.x] == connected_V[node_i.y, node_i.x]:
            end_i.matchPoint(node_i.x, node_i.y)
            node_i.matchOnePoint(end_i.x,end_i.y,"E")
            break
    if end_i.xm == -1:
        for start_i in StartPointList:
            if connected_H[start_i.y, start_i.x] == connected_H[end_i.y, end_i.x] or connected_V[start_i.y, start_i.x] == connected_V[end_i.y, end_i.x]:
                end_i.matchPoint(start_i.x, start_i.y)
                if start_i.xm == -1:
                    start_i.matchPoint(end_i.x, end_i.y)
                break
progress_bar(70,start_time)
# 识别并去除文字
ocr_result1, ocr_result2 = word_ocr(crop)
ocr_res1_copy = copy.deepcopy(ocr_result1)
ocr_res2_copy = copy.deepcopy(ocr_result2)
progress_bar(95,start_time)
# 为圆形类绑定文字
for idx in range(len(ocr_res1_copy)):
    res = ocr_res1_copy[idx]
    for line in res:
         #print(line)
        po1 = int(line[0][0][0])
        po2 = int(line[0][0][1])
        po3 = int(line[0][2][0])
        po4 = int(line[0][2][1])
        word_x = (po1 + po3) / 2
        word_y = (po2 + po4) / 2
        for cc in CircleList:
            if abs(word_x - cc.x) < cc.r and abs(word_y - cc.y) < cc.r:
                if line[1][0] == 'MP':
                    cc.word_list.append(line)
                else:
                    cc.name = line
                ocr_result1[idx].remove(line)
                break
            elif abs(word_x - cc.x) < 150 and abs(word_y - cc.y) < 33:
                cc.word_list.append(line)
                ocr_result1[idx].remove(line)
                break
        for hex in Hexagon_List:
            cv.rectangle(crop, (hex_i.x-hex_i.r, hex_i.y-int(hex_i.r*0.9)), (hex_i.x+hex_i.r, hex_i.y+int(hex_i.r*0.9)), (255,255,255), -1)
            if abs(word_x - hex.x) < hex.r and abs(word_y - hex.y) < int(hex.r*0.86):
                hex.name = line
                ocr_result1[idx].remove(line)
                break
            elif abs(word_x - hex.x) < 150 and abs(word_y - hex.y) < 37:
                hex.word_list.append(line)
                ocr_result1[idx].remove(line)
                break
        for ea in Endarrow_List:
            if abs(word_x - ea.x) < 40 and word_y - ea.y < 30 and word_y - ea.y > 0:
                ea.name = line
                ocr_result1[idx].remove(line)
                break
            elif abs(word_x - ea.x) < 150 and abs(word_y - ea.y) < 33:
                ea.word_list.append(line)
                ocr_result1[idx].remove(line)
                break
        for pg in Playground_List:
            cv.rectangle(crop, (pg.rectangle[0], pg.rectangle[1]), (pg.rectangle[2], pg.rectangle[3]), (255,255,255), -1)
            if word_x > pg.rectangle[0] and word_x < pg.rectangle[2] and word_y > pg.rectangle[1] and word_y < pg.rectangle[3]:
                pg.name = line
                ocr_result1[idx].remove(line)
                break
            elif abs(word_x - pg.x) < 150 and abs(word_y - pg.y) < 33:
                pg.word_list.append(line)
                ocr_result1[idx].remove(line)
                break
for idx in range(len(ocr_res2_copy)):
    res = ocr_res2_copy[idx]
    for line in res:
         #print(line)
        po1 = int(line[0][0][0])
        po2 = int(line[0][0][1])
        po3 = int(line[0][2][0])
        po4 = int(line[0][2][1])
        word_x = (po1 + po3) / 2
        word_y = (po2 + po4) / 2
        for cc in CircleList:
            if abs(word_x - cc.x) < cc.r and abs(word_y - cc.y) < cc.r:
                if line[1][0] == 'MP':
                    cc.word_list.append(line)
                else:
                    cc.name = line
                ocr_result2[idx].remove(line)
                break
            elif abs(word_x - cc.x) < 150 and abs(word_y - cc.y) < 33:
                cc.word_list.append(line)
                ocr_result2[idx].remove(line)
                break
        for hex in Hexagon_List:
            if abs(word_x - hex.x) < hex.r and abs(word_y - hex.y) < int(hex.r*0.86):
                hex.name = line
                ocr_result1[idx].remove(line)
                break
            elif abs(word_x - hex.x) < 150 and abs(word_y - hex.y) < 37:
                hex.word_list.append(line)
                ocr_result1[idx].remove(line)
                break
        for ea in Endarrow_List:
            if abs(word_x - ea.x) < 40 and word_y - ea.y < 30 and word_y - ea.y > 0:
                ea.name = line
                ocr_result1[idx].remove(line)
                break
            elif abs(word_x - ea.x) < 150 and abs(word_y - ea.y) < 33:
                ea.word_list.append(line)
                ocr_result1[idx].remove(line)
                break
        for pg in Endarrow_List:
            if word_x > pg.rectangle[0] and word_x < pg.rectangle[2] and word_y > pg.rectangle[1] and word_y < pg.rectangle[3]:
                pg.name = line
                ocr_result1[idx].remove(line)
                break
            elif abs(word_x - pg.x) < 150 and abs(word_y - pg.y) < 33:
                pg.word_list.append(line)
                ocr_result1[idx].remove(line)
                break

# 合并所有圆形类List
CircleList = CircleList + Endarrow_List + Playground_List + Hexagon_List
# cv.imwrite("ls.png",crop)
# 识别线段
s_lines_narray, d_lines_narray = line_detect(crop)
if not s_lines_narray is None:
    s_lines = s_lines_narray.tolist()
else:
    s_lines = []
if not d_lines_narray is None:
    d_lines = d_lines_narray.tolist()
else:
    d_lines = []
#lines = np.concatenate((s_lines, d_lines), axis = 0)
#lines_detect_test_pic = copy.deepcopy(crop_org)

# 二次判断，将虚线中实际为实线的部分并入实线，将实线中实际为虚线的部分并入虚线
def line_classify(if_real, threshold, threshold_linedt):
    if if_real:
        sord_lines = s_lines
    else:
        sord_lines = d_lines
    for i in range(len(sord_lines)-1,-1,-1):
        line = sord_lines[i]
        x1, y1, x2, y2 = line[0]
        gray_crop = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        ret,thresh_crop = cv.threshold(gray_crop,threshold_linedt,255,cv.THRESH_BINARY_INV)
        if x2 != x1 and abs((y2-y1)/(x2-x1)) < 1:
            xs = min(x1,x2)
            xl = max(x1,x2)
            ys = y1
            yl = y1
            length = xl -xs + 1
        else:
            xs = x1
            xl = x1
            ys = min(y1,y2)
            yl = max(y1,y2)
            length = yl -ys + 1
        thresh_line = thresh_crop[ys:yl+1,xs:xl+1]
        # 统计非零元素个数
        nonzero = cv.countNonZero(thresh_line)
        # 计算直线的密度，即非零元素个数除以长度
        density = nonzero / length
        # 根据密度的大小判断是实线还是虚线
        #print(density)
        if if_real:
            if density < threshold:
                sord_lines.pop(i)
                d_lines.append(line)
        else:
            if density > threshold:
                sord_lines.pop(i)
                s_lines.append(line)

line_classify(False, 0.9, 170)
line_classify(True, 0.7, 210)


SolidHorizontalLineStartList = []
SolidHorizontalLineEndList = []
SolidVerticalLineStartList = []
SolidVerticalLineEndList = []
DottedHorizontalLineStartList = []
DottedHorizontalLineEndList = []
DottedVerticalLineStartList = []
DottedVerticalLineEndList = []

def line_type(if_real):
    if if_real:
        return s_lines, SolidHorizontalLineStartList, SolidHorizontalLineEndList, SolidVerticalLineStartList, SolidVerticalLineEndList
    else:
        return d_lines, DottedHorizontalLineStartList, DottedHorizontalLineEndList, DottedVerticalLineStartList, DottedVerticalLineEndList

def line_filter(if_real):
    lines_org, HorizontalLineStartList, HorizontalLineEndList, VerticalLineStartList, VerticalLineEndList = line_type(if_real)
    for line in lines_org: 
            x1, y1, x2, y2 = line[0]
            #print(line[0])
            #cv.line(lines_detect_test_pic, (x1, y1), (x2, y2), (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 1)
            if x2 != x1 and abs((y2-y1)/(x2-x1)) < 0.5:
                if x2 > x1:
                    hSx = LinePoint(x1,y1,x2,y2,"HS",if_real)
                    HorizontalLineStartList.append(hSx)
                    hEx = LinePoint(x2,y2,x1,y1,"HE",if_real)
                    HorizontalLineEndList.append(hEx) 
                else:
                    hSx = LinePoint(x2,y2,x1,y1,"HS",if_real)
                    HorizontalLineStartList.append(hSx)
                    hEx = LinePoint(x1,y1,x2,y2,"HE",if_real)
                    HorizontalLineEndList.append(hEx)
            elif x2 == x1 or abs((y2-y1)/(x2-x1)) > 2:
                if y2 > y1:
                    vSx = LinePoint(x1,y1,x2,y2,"VS",if_real)
                    VerticalLineStartList.append(vSx)
                    vEx = LinePoint(x2,y2,x1,y1,"VE",if_real)
                    VerticalLineEndList.append(vEx)
                else:
                    vSx = LinePoint(x2,y2,x1,y1,"VS",if_real)
                    VerticalLineStartList.append(vSx)
                    vEx = LinePoint(x1,y1,x2,y2,"VE",if_real)
                    VerticalLineEndList.append(vEx)

    # 删去相邻重复线段
    for hS1 in HorizontalLineStartList:
        for i in range(len(HorizontalLineStartList)-1,-1,-1):
            hS2 = HorizontalLineStartList[i]
            if abs(hS1.y - hS2.y) == 1 and abs(hS1.x - hS2.x) < 3:
                for j in range(len(HorizontalLineEndList)-1,-1,-1):
                    hE2 = HorizontalLineEndList[j]
                    if hE2.lx == hS2.x and hE2.ly == hS2.y:
                        HorizontalLineEndList.pop(j)
                        break
                HorizontalLineStartList.pop(i)

    for hE1 in HorizontalLineEndList:
        for i in range(len(HorizontalLineEndList)-1,-1,-1):
            hE2 = HorizontalLineEndList[i]
            if abs(hE1.y - hE2.y) == 1 and abs(hE1.x - hE2.x) < 3:
                for j in range(len(HorizontalLineStartList)-1,-1,-1):
                    hS2 = HorizontalLineStartList[j]
                    if hS2.lx == hE2.x and hS2.ly == hE2.y:
                        HorizontalLineStartList.pop(j)
                        break
                HorizontalLineEndList.pop(i)

    if if_real:
        (b ,g ,r) = (255, 0, 0)
    else:
        (b ,g ,r) = (0, 0, 255)
    '''for hS in HorizontalLineStartList:
        cv.line(lines_detect_test_pic, (hS.x, hS.y), (hS.lx, hS.ly), (b ,g ,r), 1)'''

    for vS1 in VerticalLineStartList:
        for i in range(len(VerticalLineStartList)-1,-1,-1):
            vS2 = VerticalLineStartList[i]
            if abs(vS1.x - vS2.x) == 1 and abs(vS1.y - vS2.y) < 3:
                for j in range(len(VerticalLineEndList)-1,-1,-1):    
                    vE2 = VerticalLineEndList[j]
                    if vE2.lx == vS2.x and vE2.ly == vS2.y:
                        VerticalLineEndList.pop(j)
                        break
                VerticalLineStartList.pop(i)

    for vE1 in VerticalLineEndList:
        for i in range(len(VerticalLineEndList)-1,-1,-1):
            vE2 = VerticalLineEndList[i]
            if abs(vE1.x - vE2.x) == 1 and abs(vE1.y - vE2.y) < 3:
                for j in range(len(VerticalLineStartList)-1,-1,-1):    
                    vS2 = VerticalLineStartList[j]
                    if vS2.lx == vE2.x and vS2.ly == vE2.y:
                        VerticalLineStartList.pop(j)
                        break
                VerticalLineEndList.pop(i)

    '''for vS in VerticalLineStartList:
        cv.line(lines_detect_test_pic, (vS.x, vS.y), (vS.lx, vS.ly), (b ,g ,r), 1)'''


line_filter(True)
line_filter(False)
#cv.imwrite('lines_detect_test_pic.png', lines_detect_test_pic)
# 配对函数
def match_Node(Pcurrent,range_i):
    range_Node = min(max(int(range_i/5),10),15)
    mindis2 = 9999 
    for ma in NodeList:
        if ma.x == Pcurrent.x and ma.y == Pcurrent.y:
            continue
        dis_x = abs(ma.x - Pcurrent.x)
        dis_y = abs(ma.y - Pcurrent.y)
        if dis_x < range_Node and dis_y < range_Node:
            if dis_x**2 + dis_y**2 <= mindis2:
                mindis2 = dis_x**2 + dis_y**2
                Pmatched = ma
        if mindis2 < 9999:
            return Pmatched

def match_Start(Pcurrent,range_i):
    range_Start = min(max(int(range_i/5),4),15)
    mindis2 = 9999 
    for ma in StartPointList:
        if ma.xm != -1:
            continue
        dis_x = Pcurrent.x - ma.x
        dis_y = abs(ma.y - Pcurrent.y)
        if dis_x > -range_Start and dis_x < 2*range_Start and dis_y < range_Start:
            if dis_x**2 + dis_y**2 <= mindis2:
                mindis2 = dis_x**2 + dis_y**2
                Pmatched = ma
    if mindis2 < 9999:
        return Pmatched   

def match_End(Pcurrent,range_i):
    range_End = min(max(int(range_i/5),4),15)
    mindis2 = 9999
    for ma in EndPointList:
        if ma.xm != -1:
            continue
        dis_x = ma.x - Pcurrent.x
        dis_y = abs(ma.y - Pcurrent.y)
        if dis_x > -range_End and dis_x < 2*range_End and dis_y < range_End:
            if dis_x**2 + dis_y**2 <= mindis2:
                mindis2 = dis_x**2 + dis_y**2
                Pmatched = ma
    if mindis2 < 9999:
        return Pmatched
     
def match_HorizontalLineStart(Pcurrent, Ppre,range_i, if_real):
    lines_org, HorizontalLineStartList, HorizontalLineEndList, VerticalLineStartList, VerticalLineEndList = line_type(if_real)
    mindis2 = 99999
    for ma in HorizontalLineStartList:
        pre_flag = False
        for pre in Ppre:
            if (pre.x == ma.x and pre.y == ma.y) or (pre.x == ma.lx and pre.y == ma.ly):
                pre_flag = True
                break
        if pre_flag or (Pcurrent.x == ma.lx and Pcurrent.y == ma.ly):
            continue
        dis_x = ma.x - Pcurrent.x
        if if_real:
            if ((abs(ma.y - Pcurrent.y) < int(range_i/2) and dis_x > -int(range_i/2) and dis_x < range_i and 
                (connected_H[ma.y, ma.x] == connected_H[Pcurrent.y, Pcurrent.x] or connected_V[ma.y, ma.x] == connected_V[Pcurrent.y, Pcurrent.x])) or
                (abs(ma.y - Pcurrent.y) < int(range_i/10) and dis_x > -int(range_i/10) and dis_x < int(range_i/5))):
                dis2 = dis_x**2 + (ma.y - Pcurrent.y)**2
                if dis2 < mindis2:
                    mindis2 = dis2
                    Pmatched = ma
        else:
            if abs(ma.y - Pcurrent.y) < int(range_i/4) and dis_x > -int(range_i/4) and dis_x < int(range_i/2):
                dis2 = dis_x**2 + (ma.y - Pcurrent.y)**2
                if dis2 < mindis2:
                    mindis2 = dis2
                    Pmatched = ma

    if mindis2 < 99999:
        for mam in HorizontalLineEndList:
            if mam.x == Pmatched.lx and mam.y == Pmatched.ly:
                return mam

def match_HorizontalLineEnd(Pcurrent, Ppre,range_i, if_real):
    lines_org, HorizontalLineStartList, HorizontalLineEndList, VerticalLineStartList, VerticalLineEndList = line_type(if_real)
    mindis2 = 99999 
    for ma in HorizontalLineEndList:
        pre_flag = False
        for pre in Ppre:
            if (pre.x == ma.x and pre.y == ma.y) or (pre.x == ma.lx and pre.y == ma.ly):
                pre_flag = True
                break
        if pre_flag or (Pcurrent.x == ma.lx and Pcurrent.y == ma.ly):
            continue
        dis_x = Pcurrent.x - ma.x
        if if_real:
            if ((abs(ma.y - Pcurrent.y) < int(range_i/2) and dis_x > -int(range_i/2) and dis_x < range_i and 
                (connected_H[ma.y, ma.x] == connected_H[Pcurrent.y, Pcurrent.x] or connected_V[ma.y, ma.x] == connected_V[Pcurrent.y, Pcurrent.x])) or 
                (abs(ma.y - Pcurrent.y) < int(range_i/10) and dis_x > -int(range_i/10) and dis_x < int(range_i/5))):
                dis2 = dis_x**2 + (ma.y - Pcurrent.y)**2
                if dis2 < mindis2:
                    mindis2 = dis2
                    Pmatched = ma
        else:
            if abs(ma.y - Pcurrent.y) < int(range_i/4) and dis_x > -int(range_i/4) and dis_x < int(range_i/2):
                dis2 = dis_x**2 + (ma.y - Pcurrent.y)**2
                if dis2 < mindis2:
                    mindis2 = dis2
                    Pmatched = ma
    if mindis2 < 99999:
        for mam in HorizontalLineStartList:
            if mam.x == Pmatched.lx and mam.y == Pmatched.ly:
                return mam
            
def match_VerticalLineEnd(Pcurrent, Ppre,range_i, if_real):
    lines_org, HorizontalLineStartList, HorizontalLineEndList, VerticalLineStartList, VerticalLineEndList = line_type(if_real)
    mindis2 = 99999
    for ma in VerticalLineEndList:
        pre_flag = False
        for pre in Ppre:
            if (pre.x == ma.x and pre.y == ma.y) or (pre.x == ma.lx and pre.y == ma.ly):
                pre_flag = True
                break
        if pre_flag or (Pcurrent.x == ma.lx and Pcurrent.y == ma.ly):
            continue
        dis_y = Pcurrent.y - ma.y
        if if_real:
            if ((abs(ma.x - Pcurrent.x) < int(range_i/2) and dis_y > -int(range_i/2) and dis_y < range_i and 
                (connected_H[ma.y, ma.x] == connected_H[Pcurrent.y, Pcurrent.x] or connected_V[ma.y, ma.x] == connected_V[Pcurrent.y, Pcurrent.x])) or
                (abs(ma.x - Pcurrent.x) < int(range_i/10) and dis_y > -int(range_i/10) and dis_y < int(range_i/5))):
                dis2 = dis_y**2 + (ma.x - Pcurrent.x)**2
                if dis2 < mindis2:
                    mindis2 = dis2
                    Pmatched = ma
        else:
            if abs(ma.x - Pcurrent.x) < int(range_i/4) and dis_y > -int(range_i/4) and dis_y < int(range_i/2):
                dis2 = dis_y**2 + (ma.x - Pcurrent.x)**2
                if dis2 < mindis2:
                    mindis2 = dis2
                    Pmatched = ma 
    if mindis2 < 99999:
        for mam in VerticalLineStartList:
            if mam.x == Pmatched.lx and mam.y == Pmatched.ly:
                return mam

def match_VerticalLineStart(Pcurrent, Ppre,range_i, if_real):
    lines_org, HorizontalLineStartList, HorizontalLineEndList, VerticalLineStartList, VerticalLineEndList = line_type(if_real)
    mindis2 = 99999
    for ma in VerticalLineStartList:
        pre_flag = False
        for pre in Ppre:
            if (pre.x == ma.x and pre.y == ma.y) or (pre.x == ma.lx and pre.y == ma.ly):
                pre_flag = True
                break
        if pre_flag or (Pcurrent.x == ma.lx and Pcurrent.y == ma.ly):
            continue
        dis_y = ma.y - Pcurrent.y
        if if_real:
            if ((abs(ma.x - Pcurrent.x) < int(range_i/2) and dis_y > -int(range_i/2) and dis_y < range_i and 
                (connected_H[ma.y, ma.x] == connected_H[Pcurrent.y, Pcurrent.x] or connected_V[ma.y, ma.x] == connected_V[Pcurrent.y, Pcurrent.x])) or
                (abs(ma.x - Pcurrent.x) < int(range_i/10) and dis_y > -int(range_i/10) and dis_y < int(range_i/5))):
                dis2 = dis_y**2 + (ma.x - Pcurrent.x)**2
                if dis2 < mindis2:
                    mindis2 = dis2
                    Pmatched = ma
        else:
            if abs(ma.x - Pcurrent.x) < int(range_i/4) and dis_y > -int(range_i/4) and dis_y < int(range_i/2):
                dis2 = dis_y**2 + (ma.x - Pcurrent.x)**2
                if dis2 < mindis2:
                    mindis2 = dis2
                    Pmatched = ma
    if mindis2 < 99999:
        for mam in VerticalLineEndList:
            if mam.x == Pmatched.lx and mam.y == Pmatched.ly:
                return mam

# 将相连接的节点互相配对
for node_i in NodeList:
    '''if node_i.x < 1240 and node_i.x > 1220 and node_i.y < 530 and node_i.y > 510:
        print(node_i.x, node_i.y)'''
    Pcurrent = node_i 
    Ppre = []
    range_i = 5
    judged_flag = False
    # 首先判断该节点的虚实
    while True:
        if match_HorizontalLineEnd(Pcurrent, Ppre,range_i,True):
            node_i.judge_real(True)
            break
        elif match_HorizontalLineEnd(Pcurrent, Ppre,range_i,False):
            node_i.judge_real(False)
            break
        elif match_VerticalLineEnd(Pcurrent, Ppre,range_i, True):
            node_i.judge_real(True)
            break
        elif match_VerticalLineEnd(Pcurrent, Ppre,range_i, False):
            node_i.judge_real(False)
            break
        elif match_HorizontalLineStart(Pcurrent, Ppre,range_i,True):
            node_i.judge_real(True)
            break
        elif match_HorizontalLineStart(Pcurrent, Ppre,range_i,False):
            node_i.judge_real(False)
            break
        elif match_VerticalLineStart(Pcurrent, Ppre,range_i, True):
            node_i.judge_real(True)
            break
        elif match_VerticalLineStart(Pcurrent, Ppre,range_i, False):
            node_i.judge_real(False)
            break
        elif range_i <= 500:
            range_i += 5
        else:
            break

    range_i = 5
    flag_initial = True
    while True:      
        if (Pcurrent.type == "N" and flag_initial) or Pcurrent.type == "VS":
            if match_Node(Pcurrent,range_i):
                if Pcurrent.type == "N":
                    flag_initial = False
                Pcurrent = match_Node(Pcurrent,range_i)
                range_i = 5
            elif match_VerticalLineEnd(Pcurrent, Ppre,range_i, node_i.if_real):
                if Pcurrent.type == "N":
                    flag_initial = False
                Pcurrent = match_VerticalLineEnd(Pcurrent, Ppre,range_i, node_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif range_i <= 100:
                range_i += 5
            else:
                break
        elif Pcurrent.type == "VE":
            if match_Node(Pcurrent,range_i):
                Pcurrent = match_Node(Pcurrent,range_i)
                range_i = 5
            elif match_VerticalLineStart(Pcurrent, Ppre,range_i, node_i.if_real):
                Pcurrent = match_VerticalLineStart(Pcurrent, Ppre,range_i, node_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif range_i <= 100:
                range_i += 5
            else:
                break
        else:
            break
    if Pcurrent.type == "N":
        if (not (node_i.x == Pcurrent.x and node_i.y == Pcurrent.y)) and (not node_i.if_linked_with(Pcurrent.x, Pcurrent.y)):
            node_i.link_other_Node(Pcurrent.x, Pcurrent.y)
            Pcurrent.link_other_Node(node_i.x, node_i.y)

    Pcurrent = node_i 
    Ppre = []
    range_i = 5
    flag_initial = True
    while True:      
        if (Pcurrent.type == "N" and flag_initial) or Pcurrent.type == "HS":
            if match_Node(Pcurrent,range_i):
                if  Pcurrent.type == "N":
                    flag_initial = False
                Pcurrent = match_Node(Pcurrent,range_i)
                range_i = 5
            elif match_HorizontalLineEnd(Pcurrent, Ppre,range_i, node_i.if_real):
                if  Pcurrent.type == "N":
                    flag_initial = False
                Pcurrent = match_HorizontalLineEnd(Pcurrent, Ppre,range_i, node_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif range_i <= 100:
                range_i += 5
            else:
                break
        elif Pcurrent.type == "HE":
            if match_Node(Pcurrent,range_i):
                Pcurrent = match_Node(Pcurrent,range_i)
                range_i = 5
            elif match_HorizontalLineStart(Pcurrent, Ppre,range_i, node_i.if_real):
                Pcurrent = match_HorizontalLineStart(Pcurrent, Ppre,range_i, node_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif range_i <= 100:
                range_i += 5
            else:
                break
        else:
            break
    if Pcurrent.type == "N":
        if (not (node_i.x == Pcurrent.x and node_i.y == Pcurrent.y)) and (not node_i.if_linked_with(Pcurrent.x, Pcurrent.y)):
            node_i.link_other_Node(Pcurrent.x, Pcurrent.y)
            Pcurrent.link_other_Node(node_i.x, node_i.y)
                
# 为起点配对终点
for start_i in StartPointList:
    '''if start_i.x > 800 and start_i.x < 820 and start_i.y > 510 and start_i.y < 530:
        print(start_i.x,start_i.y)'''
    Pcurrent = start_i 
    Ppre = []
    range_i = 5
    while True:
        if match_HorizontalLineStart(Pcurrent, Ppre,range_i,True):
            start_i.judge_real(True)
            break
        elif match_HorizontalLineStart(Pcurrent, Ppre,range_i,False):
            start_i.judge_real(False)
            break
        elif match_VerticalLineEnd(Pcurrent, Ppre,range_i, True):
            start_i.judge_real(True)
            break
        elif match_VerticalLineEnd(Pcurrent, Ppre,range_i, False):
            start_i.judge_real(False)
            break
        elif match_VerticalLineStart(Pcurrent, Ppre,range_i, True):
            start_i.judge_real(True)
            break
        elif match_VerticalLineStart(Pcurrent, Ppre,range_i, False):
            start_i.judge_real(False)
            break
        elif range_i <= 500:
            range_i += 5
        else:
            break
    if start_i.xm != -1:
        continue
    range_i = 5
    while Pcurrent.type != "N" and Pcurrent.type != "E":
        if Pcurrent.type == "S" or Pcurrent.type == "HE": 
            if match_Node(Pcurrent,range_i):
                Pcurrent = match_Node(Pcurrent,range_i)
                range_i = 5
            elif match_HorizontalLineStart(Pcurrent, Ppre,range_i, start_i.if_real):
                Pcurrent = match_HorizontalLineStart(Pcurrent, Ppre,range_i, start_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_VerticalLineEnd(Pcurrent, Ppre,range_i, start_i.if_real):
                Pcurrent = match_VerticalLineEnd(Pcurrent, Ppre,range_i, start_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_VerticalLineStart(Pcurrent, Ppre,range_i, start_i.if_real):
                Pcurrent = match_VerticalLineStart(Pcurrent, Ppre,range_i, start_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_End(Pcurrent,range_i):
                Pcurrent = match_End(Pcurrent,range_i)
                range_i = 5
            elif range_i <= 200:
                range_i += 5
            else:
                break        
        elif Pcurrent.type == "VS":
            if match_Node(Pcurrent,range_i):
                Pcurrent = match_Node(Pcurrent,range_i)
                range_i = 5
            elif match_VerticalLineEnd(Pcurrent, Ppre,range_i, start_i.if_real):
                Pcurrent = match_VerticalLineEnd(Pcurrent, Ppre,range_i, start_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_HorizontalLineStart(Pcurrent, Ppre,range_i, start_i.if_real):
                Pcurrent = match_HorizontalLineStart(Pcurrent, Ppre,range_i, start_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_End(Pcurrent,range_i):
                Pcurrent = match_End(Pcurrent,range_i)
                range_i = 5
            elif range_i <= 200:
                range_i += 5
            else:
                break
        elif Pcurrent.type == "VE":
            if match_Node(Pcurrent,range_i):
                Pcurrent = match_Node(Pcurrent,range_i)
                range_i = 5
            elif match_VerticalLineStart(Pcurrent, Ppre,range_i, start_i.if_real):
                Pcurrent = match_VerticalLineStart(Pcurrent, Ppre,range_i, start_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_HorizontalLineStart(Pcurrent, Ppre,range_i, start_i.if_real):
                Pcurrent = match_HorizontalLineStart(Pcurrent, Ppre,range_i, start_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_End(Pcurrent,range_i):
                Pcurrent = match_End(Pcurrent,range_i)
                range_i = 5
            elif range_i <= 200:
                range_i += 5
            else:
                break
        else:
            break
    if Pcurrent.type == "N" or Pcurrent.type == "E":
        start_i.matchPoint(Pcurrent.x, Pcurrent.y)
        if Pcurrent.type == "E":
            Pcurrent.matchPoint(start_i.x, start_i.y)
        else:
            Pcurrent.matchOnePoint(start_i.x,start_i.y,"S")

# 为终点配对起点
for end_i in EndPointList:
    Pcurrent = end_i 
    Ppre = []
    range_i = 5
    while True:
        if match_HorizontalLineEnd(Pcurrent, Ppre,range_i,True):
            end_i.judge_real(True)
            break
        elif match_HorizontalLineEnd(Pcurrent, Ppre,range_i,False):
            end_i.judge_real(False)
            break
        elif match_VerticalLineEnd(Pcurrent, Ppre,range_i, True):
            end_i.judge_real(True)
            break
        elif match_VerticalLineEnd(Pcurrent, Ppre,range_i, False):
            end_i.judge_real(False)
            break
        elif match_VerticalLineStart(Pcurrent, Ppre,range_i, True):
            end_i.judge_real(True)
            break
        elif match_VerticalLineStart(Pcurrent, Ppre,range_i, False):
            end_i.judge_real(False)
            break
        elif range_i <= 500:
            range_i += 5
        else:
            break
    if end_i.xm != -1:
        continue
    range_i = 5
    while Pcurrent.type != "N" and Pcurrent.type != "S":
        if Pcurrent.type == "E" or Pcurrent.type == "HS": 
            if match_Node(Pcurrent,range_i):
                Pcurrent = match_Node(Pcurrent,range_i)
                range_i = 5
            elif match_HorizontalLineEnd(Pcurrent, Ppre,range_i, end_i.if_real):
                Pcurrent = match_HorizontalLineEnd(Pcurrent, Ppre,range_i, end_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_VerticalLineEnd(Pcurrent, Ppre,range_i, end_i.if_real):
                Pcurrent = match_VerticalLineEnd(Pcurrent, Ppre,range_i, end_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_VerticalLineStart(Pcurrent, Ppre,range_i, end_i.if_real):
                Pcurrent = match_VerticalLineStart(Pcurrent, Ppre,range_i, end_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_Start(Pcurrent,range_i):
                Pcurrent = match_Start(Pcurrent,range_i)
                range_i = 5
            elif range_i <= 200:
                range_i += 5
            else:
                break        
        elif Pcurrent.type == "VS":
            if match_Node(Pcurrent,range_i):
                Pcurrent = match_Node(Pcurrent,range_i)
                range_i = 5
            elif match_VerticalLineEnd(Pcurrent, Ppre,range_i, end_i.if_real):
                Pcurrent = match_VerticalLineEnd(Pcurrent, Ppre,range_i, end_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_HorizontalLineEnd(Pcurrent, Ppre,range_i, end_i.if_real):
                Pcurrent = match_HorizontalLineEnd(Pcurrent, Ppre,range_i, end_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_Start(Pcurrent,range_i):
                Pcurrent = match_Start(Pcurrent,range_i)
                range_i = 5
            elif range_i <= 200:
                range_i += 5
            else:
                break
        elif Pcurrent.type == "VE":
            if match_Node(Pcurrent,range_i):
                Pcurrent = match_Node(Pcurrent,range_i)
                range_i = 5
            elif match_VerticalLineStart(Pcurrent, Ppre,range_i, end_i.if_real):
                Pcurrent = match_VerticalLineStart(Pcurrent, Ppre,range_i, end_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_HorizontalLineEnd(Pcurrent, Ppre,range_i, end_i.if_real):
                Pcurrent = match_HorizontalLineEnd(Pcurrent, Ppre,range_i, end_i.if_real)
                Ppre.append(Pcurrent)
                range_i = 5
            elif match_Start(Pcurrent,range_i):
                Pcurrent = match_Start(Pcurrent,range_i)
                range_i = 5
            elif range_i <= 200:
                range_i += 5
            else:
                break
        else:
            break
    if Pcurrent.type == "N" or Pcurrent.type == "S":
        end_i.matchPoint(Pcurrent.x, Pcurrent.y)
        if Pcurrent.type == "S":
            Pcurrent.matchPoint(end_i.x, end_i.y)
        else:
            Pcurrent.matchOnePoint(end_i.x, end_i.y,"E")

# 在原图中用不同颜色的点展示识别并配对好的起点/节点/终点
for node_i in NodeList:
    cv.putText(crop_org, "N", (node_i.x, node_i.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
    if node_i.if_real == True:
        cv.putText(crop_org, "R", (node_i.x - 10, node_i.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
    else:
        cv.putText(crop_org, "I", (node_i.x - 10, node_i.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
    if node_i.numNode > 0:
        if ((crop_org[node_i.y,node_i.x][0] == 255 and crop_org[node_i.y,node_i.x][1] == 255 and crop_org[node_i.y,node_i.x][2] == 255) or 
            (crop_org[node_i.y,node_i.x][0] == 0 and crop_org[node_i.y,node_i.x][1] == 0 and crop_org[node_i.y,node_i.x][2] == 0)):
            (b, g, r) = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        else:
            b = crop_org[node_i.y,node_i.x][0]
            g = crop_org[node_i.y,node_i.x][1]
            r = crop_org[node_i.y,node_i.x][2]
        cv.circle(crop_org, (node_i.x, node_i.y), 10, (int(b), int(g), int(r)), -1)
        for (node_j_x, node_j_y) in node_i.arrNode:
            cv.circle(crop_org, (node_j_x, node_j_y), 10, (int(b), int(g), int(r)), -1)
    else:
        (b, g, r) = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv.circle(crop_org, (node_i.x, node_i.y), 10, (b, g, r), -1)

for end_i in EndPointList:
    cv.putText(crop_org, "E", (end_i.x, end_i.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    if end_i.if_real == True:
        cv.putText(crop_org, "R", (end_i.x - 10, end_i.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    else:
        cv.putText(crop_org, "I", (end_i.x - 10, end_i.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    if end_i.inv == True:
        cv.putText(crop_org, "V", (end_i.x - 20, end_i.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        
    if end_i.xm != -1:
        if ((crop_org[end_i.ym,end_i.xm][0] == 255 and crop_org[end_i.ym,end_i.xm][1] == 255 and crop_org[end_i.ym,end_i.xm][2] == 255) or 
            (crop_org[end_i.ym,end_i.xm][0] == 0 and crop_org[end_i.ym,end_i.xm][1] == 0 and crop_org[end_i.ym,end_i.xm][2] == 0)):
            (b, g, r) = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv.circle(crop_org, (end_i.x, end_i.y), 10, (b, g, r), -1)
            cv.circle(crop_org, (end_i.xm, end_i.ym), 10, (b, g, r), -1)
        else:
            b = crop_org[end_i.ym,end_i.xm][0]
            g = crop_org[end_i.ym,end_i.xm][1]
            r = crop_org[end_i.ym,end_i.xm][2]
            cv.circle(crop_org, (end_i.x, end_i.y), 10, (int(b), int(g), int(r)), -1)
        #cv.line(crop_org, (end_i.x, end_i.y), (end_i.xm, end_i.ym), (0, 200, 200), 3)
    #cv.circle(crop_org, (end_i.x, end_i.y-1), 2, (0, 255, 0), -1)

for start_i in StartPointList:
    cv.putText(crop_org, "S", (start_i.x, start_i.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    if start_i.if_real == True:
        cv.putText(crop_org, "R", (start_i.x - 10, start_i.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    else:
        cv.putText(crop_org, "I", (start_i.x - 10, start_i.y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    if start_i.xm != -1:
        if ((crop_org[start_i.ym,start_i.xm][0] == 255 and crop_org[start_i.ym,start_i.xm][1] == 255 and crop_org[start_i.ym,start_i.xm][2] == 255) or 
            (crop_org[start_i.ym,start_i.xm][0] == 0 and crop_org[start_i.ym,start_i.xm][1] == 0 and crop_org[start_i.ym,start_i.xm][2] == 0)):
            (b, g, r) = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            start_imatchword = str(start_i.xm) + " " + str(start_i.ym)
            cv.putText(crop_org, start_imatchword, (start_i.x + 10, start_i.y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            cv.circle(crop_org, (start_i.x, start_i.y), 10, (b, g, r), -1)
            cv.circle(crop_org, (start_i.xm, start_i.ym), 10, (b, g, r), -1)
        else:
            b = crop_org[start_i.ym,start_i.xm][0]
            g = crop_org[start_i.ym,start_i.xm][1]
            r = crop_org[start_i.ym,start_i.xm][2]
            cv.circle(crop_org, (start_i.x, start_i.y), 10, (int(b), int(g), int(r)), -1)
        #cv.line(crop_org, (start_i.x, start_i.y), (start_i.xm, start_i.ym), (200, 200, 0), 3)
    #cv.circle(crop_org, (start_i.x, start_i.y-1), 2, (255, 0, 0), -1)

Op_color_list = []
for Op_i in OperatorList:
    (b, g, r) = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    Op_color_list.append((Op_i.x, Op_i.y, b, g, r))
    cv.putText(crop_org, str(Op_i.num_Start), (Op_i.x, Op_i.y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (b, g, r), 2)
    cv.putText(crop_org, Op_i.optype, (Op_i.x, Op_i.y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (b, g, r), 2)
    cv.putText(crop_org, str(Op_i.num_End), (Op_i.x, Op_i.y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (b, g, r), 2)
    cv.rectangle(crop_org, (Op_i.rectangle[0], Op_i.rectangle[1]) , (Op_i.rectangle[2], Op_i.rectangle[3]), (b, g, r), 2)

# 将OCR识别到的文字绑定到起点、终点或是运算器
for idx in range(len(ocr_result1)):
    res = ocr_result1[idx]
    for word in res:
        po1 = int(word[0][0][0])
        po2 = int(word[0][0][1])
        po3 = int(word[0][2][0])
        po4 = int(word[0][2][1])
        txts = word[1][0]
        word_x = (po1 + po3)/2
        word_y = (po2 + po4)/2

        range_x = 5
        range_y = 100
        linked_flag = False
        while range_x <= 100 and (not linked_flag):
            for Op_i in OperatorList:
                dis_x = abs(word_x - Op_i.x)
                dis_y = abs(word_y - Op_i.y)
                if dis_x < range_x and dis_y < range_y:
                    Op_i.link_word(po1,po2,po3,po4,txts)
                    linked_flag = True
                    break
            range_x += 5

        range_x = 5
        range_y = 20
        while range_x <= 200 and (not linked_flag):
            if range_x >= 100:
                range_y = 40
            for ma in StartPointList:
                dis_x = abs(word_x - ma.x)
                dis_y = abs(word_y - ma.y)
                if dis_x < range_x and dis_y < range_y:
                    ma.link_word(po1,po2,po3,po4,txts)
                    linked_flag = True
                    break
            range_x += 5

        range_x = 5
        range_y = 20
        while range_x <= 200 and (not linked_flag):
            if range_x >= 100:
                range_y = 40
            for ma in EndPointList:
                dis_x = abs(word_x - ma.x)
                dis_y = abs(word_y - ma.y)
                if dis_x < range_x and dis_y < range_y:
                    ma.link_word(po1,po2,po3,po4,txts)
                    linked_flag = True
                    break
            range_x += 5
             
for idx in range(len(ocr_result2)):
    res = ocr_result2[idx]
    for word in res:
        po1 = int(word[0][0][0])
        po2 = int(word[0][0][1])
        po3 = int(word[0][2][0])
        po4 = int(word[0][2][1])
        txts = word[1][0]
        word_x = (po1 + po3)/2
        word_y = (po2 + po4)/2
        
        range_x = 5
        range_y = 20
        linked_flag = False
        while range_x <= 200 and (not linked_flag):
            if range_x >= 100:
                range_y = 40
            for ma in StartPointList:
                dis_x = abs(word_x - ma.x)
                dis_y = abs(word_y - ma.y)
                if dis_x < range_x and dis_y < range_y:
                    ma.link_word(po1,po2,po3,po4,txts)
                    linked_flag = True
                    break
            range_x += 5

        range_x = 5
        range_y = 20
        while range_x <= 200 and (not linked_flag):
            if range_x >= 100:
                range_y = 40
            for ma in EndPointList:
                dis_x = abs(word_x - ma.x)
                dis_y = abs(word_y - ma.y)
                if dis_x < range_x and dis_y < range_y:
                    ma.link_word(po1,po2,po3,po4,txts)
                    linked_flag = True
                    break
            range_x += 5


# 为运算符添加attribute属性
for op_i in OperatorList:
    if len(op_i.arr_word):
        if op_i.optype == 'T1_0s':
            for w in op_i.arr_word:
                w = w[4].replace(' ', '')
                if w[:3] == 'T1=':
                    if w[-1] == 's':
                        op_i.attribute = w[-2:]
                        break
                    elif w[-1] in digits:
                        atr = w[-1] + 's'
                        op_i.attribute = atr
                        break
        elif op_i.optype == '0s_T2':
            for w in op_i.arr_word:
                w = w[4].replace(' ', '')
                if w[:3] == 'T2=':
                    if w[-1] == 's':
                        op_i.attribute = w[-2:]
                        break
                    elif w[-1] in digits:
                        atr = w[-1] + 's'
                        op_i.attribute = atr
                        break
        elif op_i.optype == 'C':
            for w in op_i.arr_word:
                w = w[4].replace(' ', '')
                if w[:2] == 'C=':
                    if w[-1] == 'm':
                        op_i.attribute = w[2:]
                        break
        elif op_i.optype == 't':
            for w in op_i.arr_word:
                w = w[4].replace(' ', '')
                if w[:2] == 'T=':
                    if w[-1] == 'S' or 's':
                        op_i.attribute = w[-2:]
                        break
        elif op_i.optype == 'T1_T2':
            T1 = None
            T2 = None
            for w in op_i.arr_word:
                w = w[4].replace(' ', '')
                if w[:3] == 'T1=':
                    if w[-1] == 's':
                        T1 = w
                    elif w[-1] in digits:
                        T1 = w + 's'
                if w[:3] == 'T2=':
                    if w[-1] == 's':
                        T2 = w
                    elif w[-1] in digits:
                        T2 = w + 's'
            if (not T1 is None) and (not T2 is None):
                op_i.attribute = T1 + '_' + T2
                

#在原图以所绑定的点的颜色展示OCR识别到的文字
for start_i in StartPointList:
        b = crop_org[start_i.y,start_i.x][0]
        g = crop_org[start_i.y,start_i.x][1]
        r = crop_org[start_i.y,start_i.x][2]
        for (word_x1,word_y1,word_x2,word_y2,word_content) in start_i.arr_word:
            cv.putText(crop_org, word_content, (word_x1, word_y1), cv.FONT_HERSHEY_SIMPLEX, 0.4, (int(b), int(g), int(r)), 1)

for end_i in EndPointList:
        b = crop_org[end_i.y,end_i.x][0]
        g = crop_org[end_i.y,end_i.x][1]
        r = crop_org[end_i.y,end_i.x][2]
        for (word_x1,word_y1,word_x2,word_y2,word_content) in end_i.arr_word:
            cv.putText(crop_org, word_content, (word_x1, word_y1), cv.FONT_HERSHEY_SIMPLEX, 0.4, (int(b), int(g), int(r)), 1)

for Op_i in OperatorList:
    b = 0
    g = 0
    r = 0
    for (x, y, b_i, g_i, r_i) in Op_color_list:
        if Op_i.x == x and Op_i.y == y:
            b = b_i
            g = g_i
            r = r_i
            break
    for (word_x1,word_y1,word_x2,word_y2,word_content) in Op_i.arr_word:
            cv.putText(crop_org, word_content, (word_x1, word_y1), cv.FONT_HERSHEY_SIMPLEX, 0.4, (b ,g ,r), 1)
            # print(word_content)

# 在原图显示识别到的4圆角功能块
for rrc in RoundedRCList:
    if rrc.type == 1:
        cv.putText(crop_org, "1", (int(rrc.title[0][0][0]-20), int(rrc.title[0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv.putText(crop_org, str(rrc.title[1][0]), (int(rrc.title[0][0][0]), int(rrc.title[0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        for i in range(len(rrc.word_list)):
            cv.putText(crop_org, str(rrc.word_list[i][1][0]), (int(rrc.word_list[i][0][0][0]), int(rrc.word_list[i][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            cv.putText(crop_org, str(i), (int(rrc.word_list[i][0][0][0]-20), int(rrc.word_list[i][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    else:
        cv.putText(crop_org, str(rrc.type), (int(rrc.title[0][0][0]-20), int(rrc.title[0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv.putText(crop_org, str(rrc.title[1][0]), (int(rrc.title[0][0][0]), int(rrc.title[0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv.putText(crop_org, str(rrc.middle[1][0]), (int(rrc.middle[0][0][0]), int(rrc.middle[0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1)
        for i in range(len(rrc.left_list)):
            cv.putText(crop_org, str(rrc.left_list[i][1][0]), (int(rrc.left_list[i][0][0][0]), int(rrc.left_list[i][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
            cv.putText(crop_org, str(i), (int(rrc.left_list[i][0][0][0]-20), int(rrc.left_list[i][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        for i in range(len(rrc.right_list)):
            cv.putText(crop_org, str(rrc.right_list[i][1][0]), (int(rrc.right_list[i][0][0][0]), int(rrc.right_list[i][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
            cv.putText(crop_org, str(i), (int(rrc.right_list[i][0][0][0]-20), int(rrc.right_list[i][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
        for i in range(len(rrc.bottom_list)):
            cv.putText(crop_org, str(rrc.bottom_list[i][1][0]), (int(rrc.bottom_list[i][0][0][0]), int(rrc.bottom_list[i][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            cv.putText(crop_org, str(i), (int(rrc.bottom_list[i][0][0][0]-20), int(rrc.bottom_list[i][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            
# 在原图显示识别到的2圆角功能块
for r2rc in RoundedTwoRCList:
    cv.putText(crop_org, str(r2rc.title[1][0]), (int(r2rc.title[0][0][0]), int(r2rc.title[0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
    for i in range(len(r2rc.word_list)):
        # print(r2rc.word_list)
        cv.putText(crop_org, str(r2rc.word_list[i][0][1][0]), (int(r2rc.word_list[i][0][0][0][0]), int(r2rc.word_list[i][0][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        cv.putText(crop_org, str(i), (int(r2rc.word_list[i][0][0][0][0]-20), int(r2rc.word_list[i][0][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

# 在原图显示识别到的矩形功能块
for rab in RectangleABList:
    cv.putText(crop_org, str(rab.title[1][0]), (int(rab.title[0][0][0]), int(rab.title[0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
    for i in range(len(rab.left_list)):
        cv.putText(crop_org, str(rab.left_list[i][0][1][0]), (int(rab.left_list[i][0][0][0][0]), int(rab.left_list[i][0][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        cv.putText(crop_org, str(i), (int(rab.left_list[i][0][0][0][0]-20), int(rab.left_list[i][0][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    for i in range(len(rab.right_list)):
        cv.putText(crop_org, str(rab.right_list[i][0][1][0]), (int(rab.right_list[i][0][0][0][0]), int(rab.right_list[i][0][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        cv.putText(crop_org, str(i), (int(rab.right_list[i][0][0][0][0]-20), int(rab.right_list[i][0][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

# 在原图显示识别到的圆形
for cc in CircleList:
    (b, g, r) = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    if not cc.name is None:
        cv.putText(crop_org, str(cc.name[1][0]), (int(cc.name[0][0][0]), int(cc.name[0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (b ,g ,r), 1)
    for i in range(len(cc.word_list)):
        cv.putText(crop_org, str(cc.word_list[i][1][0]), (int(cc.word_list[i][0][0][0]), int(cc.word_list[i][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.3, (b, g, r), 1)
        cv.putText(crop_org, str(i), (int(cc.word_list[i][0][0][0]-20), int(cc.word_list[i][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 0.3, (b, g, r), 1)
output_path = './output/' + args.output
cv.imwrite(output_path,crop_org)
progress_bar(100,start_time)