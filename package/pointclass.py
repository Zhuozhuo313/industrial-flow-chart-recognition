class AnyPoint:
    def __init__(self,x,y,type):
        self.x = x
        self.y = y
        self.type = type

class SorEPoint(AnyPoint):
    def __init__(self,x,y,type):
        AnyPoint.__init__(self,x,y,type)
        self.ifMatch = False
        self.xm = -1
        self.ym = -1
        self.num_word = 0
        self.arr_word = []
        self.if_real = True
        self.inv = False

    def matchPoint(self,xm,ym):
        self.xm = xm
        self.ym = ym
        self.ifMatch = True

    def link_word(self,word_x1,word_y1,word_x2,word_y2,word_content):
        self.num_word += 1
        self.arr_word.append((word_x1,word_y1,word_x2,word_y2,word_content))

    def jgMatch(self):
        return self.ifMatch
    
    def judge_real(self, if_real):
        self.if_real = if_real
    
    def label_inv(self):
        self.inv = True

class Node(AnyPoint):
    def __init__(self,x,y):
        AnyPoint.__init__(self,x,y,"N")
        self.numStart = 0
        self.numEnd = 0
        self.numNode = 0
        self.arrStart = []
        self.arrEnd = []
        self.arrNode = []
        self.if_real = True

    def link_other_Node(self,xm,ym):
        self.numNode += 1
        self.arrNode.append((xm,ym))

    def matchOnePoint(self,xm,ym,typem):
        if typem == "S":
            self.numStart += 1
            self.arrStart.append((xm,ym))
        elif typem == "E":
            self.numEnd += 1
            self.arrEnd.append((xm,ym))
        else:
            print("illegal type!")

    def if_linked_with(self,xm,ym):
        linked = False
        for (x_i,y_i) in self.arrNode:
            if x_i == xm and y_i == ym:
                linked = True
                break
        return linked
    
    def judge_real(self, if_real):
        self.if_real = if_real

class OpPoint(AnyPoint):
    def __init__(self,x,y,optype):
        AnyPoint.__init__(self,x,y,"O")
        self.optype = optype
        self.num_Start = 0
        self.num_End = 0
        self.num_word = 0
        self.arr_Start = []
        self.arr_End = []
        self.arr_word = []
        self.rectangle = []
        self.if_triangle = False

    def link_word(self, word_x1,word_y1,word_x2,word_y2,word_content):
        self.num_word += 1
        self.arr_word.append((word_x1,word_y1,word_x2,word_y2,word_content))

    def link_SorE(self,xm,ym,typem):
        if typem == "S":
            self.num_Start += 1
            self.arr_Start.append((xm,ym))
        elif typem == "E":
            self.num_End += 1
            self.arr_End.append((xm,ym))

    def rect_inf(self,x1,y1,x2,y2):
        self.rectangle.append(x1)
        self.rectangle.append(y1)
        self.rectangle.append(x2)
        self.rectangle.append(y2)

class LinePoint(AnyPoint):
    def __init__(self,x1,y1,x2,y2,type,if_real):
        AnyPoint.__init__(self,x1,y1,type)
        self.lx = x2
        self.ly = y2    
        self.if_real = if_real