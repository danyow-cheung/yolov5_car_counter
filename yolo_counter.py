import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import time 


'''
使用yolov5的onxx格式来跑目标检测
https://github.com/guptavasu1213/Yolo-Vehicle-Counter/blob/master/yolo_video.py

'''

class Yolo_detect():
    '''定义超参数'''
    def __init__(self):
        
        # Constants.
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        self.SCORE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.45
        self.CONFIDENCE_THRESHOLD = 0.45
        
        # Text parameters.
        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.7
        self.THICKNESS = 1
        
        # Colors.
        self.BLACK  = (0,0,0)
        self.BLUE   = (255,178,50)
        self.YELLOW = (0,255,255)
        '''加载类别名'''
        classesFile = "yolo/labels.txt"

        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        # print(f'self.classes={self.classes}')
        '''
        使用yolov5s.pt 转换.onnx和直接下载onnx的模型数据类型不同
        https://forum.opencv.org/t/error-when-reading-yolo5-as-onnx-using-cv2/11507/3
        `--opset 12`添加倒出参数
        '''
        modelWeights = "yolo/yolov5s.onnx"

        #计数器
        self.count = 0 
        # 中间线设置
        self.middle_line_position = 400
        # 上下范围区间
        self.up_line_position = self.middle_line_position -100
        self.down_line_position = self.middle_line_position + 100 
       
        self.net = cv2.dnn.readNet(modelWeights)
        


    '''绘制类别'''
    def draw_label(self,img,label,x,y):
        text_size = cv2.getTextSize(label,self.FONT_FACE,self.FONT_SCALE,self.THICKNESS)
        dim,baseline = text_size[0],text_size[1]
        cv2.rectangle(img,(x,y),(x+dim[0],y+dim[1]+baseline),(0,0,0),cv2.FILLED)
        cv2.putText(img,label,(x,y+dim[1]),self.FONT_FACE,self.FONT_SCALE,self.YELLOW,self.THICKNESS)

    
    
    def pre_process(self,input_image,net):
        '''
        预处理
        将图像和网络作为参数。
        - 首先，图像被转换为​​ blob。然后它被设置为网络的输入。
        - 该函数getUnconnectedOutLayerNames()提供输出层的名称。
        - 它具有所有层的特征，图像通过这些层向前传播以获取检测。处理后返回检测结果。
        '''    
        blob = cv2.dnn.blobFromImage(input_image,1/255,(self.INPUT_HEIGHT,self.INPUT_WIDTH),[0,0,0], 1, crop=False)
        # Sets the input to the network.
        net.setInput(blob)
        # Run the forward pass to get output of the output layers.
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        return outputs
    


    

    
    def post_process(self,input_image,outputs):
        '''后处理
        过滤 YOLOv5 模型给出的良好检测
        步骤
        - 循环检测。
        - 过滤掉好的检测。
        - 获取最佳班级分数的索引。
        - 丢弃类别分数低于阈值的检测。
        '''
            
        class_ids = []
        confidences = []
        boxes = []
        rows = outputs[0].shape[1]
        
        image_height ,image_width = input_image.shape[:2]
        
        x_factor = image_width/self.INPUT_WIDTH
        y_factor = image_height/self.INPUT_HEIGHT
        

        # 循环检测
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            if confidence>self.CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)
                if (classes_scores[class_id]>self.SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx,cy,w,h = row[0],row[1],row[2],row[3]
                    left = int((cx-w/2)*x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)
        

        '''非极大值抑制来获取一个标准框'''
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        # 得到想要的
        # girl i want it i got it 
        res_box = []
        res_label = []
        res_conf = []
        
        for i in range(len(indices)): # 矩阵中通过idx可以拿到id
            # print(f'No = {i},NMS Num = {indices[i]}')

            box = boxes[indices[i]]
            
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]             
            # 描绘标准框
            cv2.rectangle(input_image, (left, top), (left + width, top + height),self.BLUE, 3*self.THICKNESS)
            
            res_box.append([left,top,width,height])
            # print(class_ids[i])
            # res_label.append(self.classes[class_ids[i]])
            res_label.append(class_ids[i])

            res_conf.append(confidences[i])

        # return res_box,res_conf,res_label,input_image
        return res_box,res_conf,res_label


    
    '''输入图片路径进行检测'''
    def detect_single_frame(self,path):

        frame = cv2.imread(path)
        # Process image.
        detections =self.pre_process(frame, self.net)
        img= self.post_process(frame.copy(), detections)
        """
        效率信息。函数getPerfProfile返回推理的总时间(t)以及每个层的时间(在layertimes中)。
        """
        t, _ = self.net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 /  cv2.getTickFrequency())
        cv2.putText(img, label, (20, 40), self.FONT_FACE, self.FONT_SCALE,  (0, 0, 255), self.THICKNESS, cv2.LINE_AA)
        
        # 展示图片
        plt.figure(figsize=(15,5))
        plt.imshow(img)
        plt.show()

    def detect(self,frame):
        '''输入图片帧进行检测，返回self.post_process里面的内容'''
        detections =self.pre_process(frame, self.net)
        res= self.post_process(frame, detections)

        return res




if __name__ == '__main__':

    path = [
        './sequence/data_frame0.jpg',
        'yolo/test_car.jpeg',
    ]

    detector = Yolo_detect()

    # image = cv2.imread(path[1])
    # res = detector.detect(image)  

    # print(res)
   
    # # # print(res.shape) 
    # length = len(res[0])
    # # for box in bbox:
    # for i in range(length) :
    #     # 像素中心坐标
    #     print(res[0][i],res[2][i])

    # 20230730 現在要測試在第一幀畫面中是不是12個object被檢測到
    # 確實是12個object，所以對應到n_init應該是單個的predict和update? maybe
    
    # cap = cv2.VideoCapture('carInHighway.mp4')
    # success,frame = cap.read()
    # while success:
    #     ret,frame = cap.read()
    #     res = detector.detect(frame)
    #     plt.imshow(res[-1])
    #     plt.show()





        

    




    
