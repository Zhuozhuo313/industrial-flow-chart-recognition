## 环境准备（Windows系统）
### Step 1：安装opencv的python库
**pip install opencv-python**
### Step 2：（可选，使用GPU加速OCR过程的推理）安装CUDA
支持CUDA版本：12.0/11.8/11.7/11.6/11.2/10.2
CUDA 11.7下载地址：https://developer.nvidia.com/cuda-11-7-0-download-archive
### Step 3：安装PaddlePaddle
从这个网站获取安装命令：https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html
### Step 4：安装PaddleOCR
**pip install "paddleocr>=2.0.1"**

## 无损提取pdf中的图像可用extract.py
将需要被提取的pdf文件放入original_pdf文件夹  
输出图片将存入extract_img文件夹


```python
# 示例
!python extract.py -i "test.pdf"
```

**-i：输入pdf的文件名（带后缀）**

## 使用教程
将需要进行推理的图片放入input文件夹  
输出图片将存入output文件夹


```python
# 示例
!python main.py -i "test_280.png" -o "test_280_output.png"
```

**-i：输入图片的文件名（带后缀）**  
**-o：输出图片的文件名（带后缀）**

## 批量处理
使用run_in_batches.py  
将需要进行推理的图片放入input文件夹，命名规则为"前缀_序号.png"  
输出图片将存入output文件夹，命名规则为"前缀_序号_后缀.png"


```python
# 示例
!python run_in_batches.py -i "test" -o "output" -s 40 -e 430
```

**-i：输入图片的文件名前缀**  
**-o：输出图片的文件名后缀**  
**-s：输入图片起始序号**  
**-e：输入图片终止序号**

## 程序思路介绍
分别找到逻辑图的每一个（线段的）起点、终点、节点，然后为每一个起点配对唯一的终点（或者起点），每一个终点配对唯一的起点（或者起点），相连的节点之间相互配对  
运算符匹配输入其的终点及其输出的起点  
识别到的文字按照规则配对到起点、终点或者运算符

## 程序输出内容介绍
### main.py文件将生成8个list，分别是：
### StartPointList：所有起点的集合。其中的成员具有如下属性：
    ifMatch：bool类型，被配对则为True，否则为False
    x：横坐标
    y：纵坐标
    xm：其配对的终点或节点的横坐标
    ym：其配对的终点或节点的纵坐标
    type：点的类型，固定为字符"S"
    num_word：与其绑定的的文字的数量（不是文字的字符数量，可以理解为单词数量）
    arr_word：list类型，其中包含所有被其绑定的文字的坐标及文字内容，其成员为如下格式：
        (word_x1,word_y1,word_x2,word_y2,word_content) 分别为（左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标，文字内容）
### EndPointList：所有终点的集合。其中的成员具有如下属性：
    ifMatch：bool类型，被配对则为True，否则为False
    x：横坐标
    y：纵坐标
    xm：其配对的起点或节点的横坐标
    ym：其配对的起点或节点的纵坐标
    type：点的类型，固定为字符"E"
    num_word：与其绑定的的文字的数量（不是文字的字符数量，可以理解为单词数量）
    arr_word：list类型，其中包含所有被其绑定的文字的坐标及文字内容，其成员为如下格式：
        (word_x1,word_y1,word_x2,word_y2,word_content) 分别为（左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标，文字内容）
### NodeList：所有节点的集合。其中的成员具有如下属性：
    x：横坐标
    y：纵坐标
    type：点的类型，固定为字符"N"
    numNode：与其相连的节点的个数
    arrNode：list类型，其中包含所有与其相连的节点的坐标，其成员为如下格式：
        (xm,ym) 分别为（与其相连的节点的横坐标，与其相连的节点的纵坐标）
**以下（numStart、numEnd、arrStart、arrEnd）为新增属性**

    numStart：与其相连的起点的个数
    numEnd：与其相连的终点的个数
    arrStart：list类型，其中包含所有与其相连的起点的坐标，其成员为如下格式：
        (xm,ym) 分别为（与其相连的起点的横坐标，与其相连的起点的纵坐标）
    arrEnd：list类型，其中包含所有与其相连的终点的坐标，其成员为如下格式：
        (xm,ym) 分别为（与其相连的终点的横坐标，与其相连的终点的纵坐标）
### OperatorList：所有运算符的集合。其中的成员具有如下属性：
    x：横坐标
    y：纵坐标
    type：点的类型，固定为字符"O"
    optype：str类型，表示运算符的类型
    num_Start，与其配对的起点的数量
    num_End，与其配对的终点的数量
    num_word，与其配对的文字的数量（不是文字的字符数量，可以理解为单词数量）
    arr_Start，list类型，其中包含所有与其配对的起点的坐标，其成员为如下格式：
        (xm,ym) 分别为（与其配对的起点的横坐标，与其配对的起点的纵坐标）
    arr_End，list类型，其中包含所有与其配对的终点的坐标，其成员为如下格式：
        (xm,ym) 分别为（与其配对的终点的横坐标，与其配对的终点的纵坐标）
    arr_word，list类型，其中包含所有与其配对的文字的坐标及文字内容，其成员为如下格式：
        (word_x1,word_y1,word_x2,word_y2,word_content)分别为（左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标，文字内容）
**新增属性**

    attribute，str类型，部分运算符具有，描述该运算符的关键属性，例如T1=5s，则attribute="5s"
    if_triangle，bool类型，该运算符右下角是否有黑色三角，有则为True，无则为False

**具体optype请查看./package/operatorkeyword.py，其中字典的key即为optype（此外还有inv表示反转器）**

## **以下为新增list**

### 首先解释一种数组，是OCR得到的文字的格式，将其记作line类型
    line[1][0]为str类型，是文字内容
    int(line[0][0][0])为该文字左上角横坐标x1
    int(line[0][0][1])为该文字左上角纵坐标y1
    int(line[0][2][0])为该文字右下角横坐标x2
    int(line[0][2][1])为该文字左上角纵坐标y2
**另外，将StartPointList/EndPointList成员的类型记作sore类型**
### RoundedRCList：4圆角矩形功能块的集合。其中的成员具有如下属性：
    x1:功能块左上角横坐标
    y1:功能块左上角纵坐标
    x2:功能块右下角横坐标
    y2:功能块右下角纵坐标
    type：int类型，取值为1、2、3。type1为单列，type2/3复杂，type2中间的字是横着的，type3中间中间的字是竖着的
    title：line类型，功能块最上面的一行字
**接下来的属性需按type分类**

    对于type == 1：
    word_list：list类型，其中包含功能块除title外的文字，成员均为line格式（有序，word_list[0]对应最上面一行的文字，依次下排）
    end_list：list类型，其中包含所有接入功能块的终点的坐标，其成员格式为：(end_i,w)
        end_i为sore类型，是接入功能块的终点
        w为line类型，是与该终点接入的部分的文字
    start_list：list类型，其中包含所有功能块接出的起点的坐标，其成员格式为：(start_i,w)，类型与end_list相同
    对于type == 2 or type ==3：
    left_list：list类型，其中包含功能块左侧的文字，成员均为line格式（有序，left_list[0]对应最上面一行的文字，依次下排）
    right_list:list类型，其中包含功能块右侧的文字，成员均为line格式（有序，right_list[0]对应最上面一行的文字，依次下排）
    bottom_list：list类型，其中包含功能块底部的文字，成员均为line格式（有序，bottom_list[0]对应最上面一行的文字，依次下排）
    middle：line类型，是该功能块中部的文字
    end_list：list类型，其中包含所有接入功能块的终点的坐标，其成员格式为：(end_i,l)
        end_i为sore类型，是接入功能块的终点
        l为line类型，是与该终点接入的功能块左侧的文字
    start_list：list类型，其中包含所有功能块接出的起点的坐标，其成员格式为：(start_i,r)
        start_i为sore类型，是功能块接出的起点
        r为line类型，是接出该起点的功能块右侧的文字

### RoundedTwoRCList：2圆角矩形功能块的集合。其中的成员具有如下属性：
    x1：功能块左上角横坐标
    y1：功能块左上角纵坐标
**x2，y2与4圆角矩形功能块中的含义不同**

    x2：功能块最上面的2圆角方块的右下角横坐标
    y2：功能块最上面的2圆角方块的右下角纵坐标
    title：line类型，功能块最上面的一行字
    word_list：list类型，包含功能块除title外的文字，成员格式为：(word,rectangle)。（该list有序，word_list[0]对应最上面一行的文字，依次下排）
        word为line类型的文字
        rectangle不需了解
        如需调用该list第i个元素，建议直接使用word_list[i-1][0]调用word部分即可
    start_list：同RoundedRCList的type1的start_list
### RectangleABList：矩形功能块的集合。其中的成员具有如下属性：
    x1：功能块左上角横坐标
    y1：功能块左上角纵坐标
**x2，y2与4圆角矩形功能块中的含义不同，与2圆角矩形功能块相同**

    x2：功能块最上面的矩形的右下角横坐标
    y2：功能块最上面的矩形的右下角纵坐标
    title：line类型，功能块最上面的一行字
    left_list：list类型，其中包含功能块左侧的文字，成员格式为：(word,rectangle)，同RoundedTwoRCList的word_list
    right_list：list类型，其中包含功能块右侧的文字，成员格式为：(word,rectangle)，同RoundedTwoRCList的word_list
    start_list：同RoundedRCList的type2/3的start_list
### CircleList：虚线圆/实线圆/六边形/终点箭头/操场形的集合。其中的成员具有如下属性：
    x：圆心横坐标
    y：圆形纵坐标
    r：圆形半径/六边形边长（type4/5该属性为-1）
    name：圆内的文字，line类型
    type：int类型，1表示虚线圆，2表示实线圆，3表示六边形，4表示终点箭头，5表示操场形
    if_start：bool类型，True表示与起点相连，False表示与终点相连
    linked_point：sore类型，与其相连的起点/终点
    word_list：list类型，与其相关的文字，内容为line类型，暂为乱序
    rectangle：list类型，仅type4/5具有，为其最小外接矩形
        rectangle[0]、rectangle[1]、rectangle[2]、rectangle[3]分别为矩形左上角横坐标、左上角纵坐标、右下角横坐标、右下角纵坐标

**以上8个list均可在main.py程序运行完成后导出到别的程序中，用于制作Demo**

## 输出图片介绍
在原图中，以圆点表示识别到的起点/终点/节点，颜色相同者表示相连  
以运算符字符类型表示识别到的运算符，字符上下的数字分别表示其输入输出的个数  
识别到的文字与配对到的起点/终点/运算符颜色相同
