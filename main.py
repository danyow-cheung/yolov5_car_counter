import os
import cv2
import numpy as np
from deep_sort import nn_matching
from deep_sort.detection import Detection
from yolo_counter import Yolo_detect
from deep_sort.tracker_danyow import Tracker


def run():
    video_path = '/Users/danyow/Desktop/yolov5_car_counter/carInHighway.mp4'
    cap = cv2.VideoCapture(video_path)
    ret,frame = cap.read()
    
    model = Yolo_detect()
    tracker = Tracker()
    while ret:
        ret,frame = cap.read()
            
        '''观察'''
        detections = []# 用来卡尔曼滤波的列表
        # 返回yolo检测器的目标xy，
        res =  model.detect(frame)
        length = len(res[0]) # 多目标，所以依次获取
        for i in range(length) :

            detections.append(Detection(tlwh=res[0][i],confidence=res[1][i],feature=res[2][i]),)
        print('total length ',len(detections))
        '''更新
        这一步应该是，使用deep_sort来跟踪了？
        > 不过deep_sort还是没有玩明白现在  
        tracker.predict()
        tracker.update(detections)
        '''
        tracker.predict()
        tracker.update(detections)

        # cv2.imshow('frame',frame)
        # if cv2.waitKey(10)==27:
        #     break

    cap.release()
    cv2.destroyAllWindows()


if __name__ =='__main__':
    run()
    # tracker = Tracker()
