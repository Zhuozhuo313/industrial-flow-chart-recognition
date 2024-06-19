import os
import fitz
import argparse

#使用fitz库无损提取pdf中的图像
#参数：file        源pdf文件完整路径（含文件名）  例如：file = 'C:/Users/86521/Desktop/test.pdf'
#参数：image_path  提取图像的保存路径            例如：image_path = 'C:/Users/86521/Desktop/Eximg/'
def ExtractImages(file,image_path):
    pdfsplit = os.path.split(file)   #分离出文件名和路径
    pdfname = pdfsplit[-1]  #获取文件名
    pdfsplit1 = os.path.splitext(pdfname)
    pdfname1 = pdfsplit1[0]   #获取不带扩展名的文件名
    # 打开pdf
    pdf = fitz.open(file)
    # 图片计数
    imgcount = 0
    # 打印pdf信息
    print("文件名:{}, 页数: {}".format(pdf, len(pdf)))

    #遍历pdf，获取每一页
    for page in pdf:
        try:
            imgcount +=1
            tupleImage = page.get_images()
            for xref in list(tupleImage):
                xref = list(xref)[0]
                img = pdf.extract_image(xref)   #获取文件扩展名，图片内容等信息
                imageFilename = ("%s." % (imgcount) + img["ext"])
                imageFilename = pdfname1 + "_" + imageFilename  #合成最终的图像的文件名
                imageFilename = os.path.join(image_path, imageFilename)   #合成最终图像完整路径名
                print(imageFilename)
                imgout = open(imageFilename, 'wb')   #byte方式新建图片
                imgout.write(img["image"])   #当前提取的图片写入磁盘
                imgout.close
        except:
            continue
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='test.pdf', required=False, help='The name of the input file.')
    args = parser.parse_args()
    file = './original_pdf/' + args.input
    #current_path = abspath(dirname(__file__))    #获取当前目录
    #pdf = os.path.join(current_path,'test.pdf')
    ExtractImages(file, './extract_img/')