class RRC_Algorithm_Block:
    def __init__(self,x1,y1,x2,y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.any_word = []
        self.title = None
        self.type = 0
        self.word_list = [] #type1使用
        self.left_list = [] #type2/3共用
        self.middle = None #type2/3共用
        self.right_list = [] #type2/3共用
        self.bottom_list = [] #type2/3共用
        self.end_list = []
        self.start_list = []

class r2rc_Algorithm_Block:
    def __init__(self,x1,y1,x2,y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.title = None
        self.word_list = []
        self.start_list = []

class rectangle_Algorithm_Block:
    def __init__(self,x1,y1,x2,y2,title):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.title = title
        self.left_list = []
        self.right_list = []
        self.start_list = []

class circle_Algorithm_Block:
    def __init__(self,x,y,r,type):
        self.x = x
        self.y = y
        self.r = r
        self.name = None
        self.type = type
        self.if_start = None
        self.linked_point = None
        self.word_list = []
        self.rectangle = []

    def rect_inf(self,x1,y1,x2,y2):
        self.rectangle.append(x1)
        self.rectangle.append(y1)
        self.rectangle.append(x2)
        self.rectangle.append(y2)