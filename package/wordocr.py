from paddleocr import PaddleOCR
import cv2

def word_ocr(img_org):
    ocr = PaddleOCR(lang="en", use_angle_cls = True, det_limit_side_len=2815, det_db_unclip_ratio=1.3, det_db_box_thresh=0.3, show_log=False)

    result = ocr.ocr(img_org)
    if len(result) == 1 and result[0] is None:
        result = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            #print(line)
            po1 = int(line[0][0][0])
            po2 = int(line[0][0][1])
            po3 = int(line[0][2][0])
            po4 = int(line[0][2][1])
            #txts = line[1][0]
            cv2.rectangle(img_org, (po1, po2), (po3, po4), (255, 255, 255),-1)
            #cv2.putText(img_org, txts, (po1, po2), cv2.FONT_HERSHEY_COMPLEX, 2, (200, 0, 200), 2)

    ocr_ch = PaddleOCR(lang="ch", use_angle_cls = True, det_limit_side_len=2815, det_db_unclip_ratio=1.3, det_db_box_thresh=0.3, show_log=False)

    result2 = ocr_ch.ocr(img_org)
    if len(result2) == 1 and result2[0] is None:
        result2 = []
    for idx in range(len(result2)):
        res = result2[idx]
        for line in res:
            #print(line)
            po1 = int(line[0][0][0])
            po2 = int(line[0][0][1])
            po3 = int(line[0][2][0])
            po4 = int(line[0][2][1])
            #txts = line[1][0]
            cv2.rectangle(img_org, (po1, po2), (po3, po4), (255, 255, 255),-1)
            #cv2.putText(img_org, txts, (po1, po2), cv2.FONT_HERSHEY_COMPLEX, 2, (200, 0, 200), 2)

    return result, result2  

def word_ocr_nocleaner(img_org):
    ocr = PaddleOCR(lang="en", use_angle_cls = True, det_limit_side_len=2815, det_db_unclip_ratio=1.8, det_db_box_thresh=0.3, show_log=False)
    result = ocr.ocr(img_org)
    if len(result) == 1 and result[0] is None:
        result = []
    return result

if __name__ == '__main__':
    img_path = 'cc.png'
    img_org = cv2.imread(img_path)
    word_ocr(img_org)
    cv2.imwrite('wd.png', img_org)