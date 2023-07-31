import os
import cv2
import numpy as np
from deep_sort import nn_matching
from deep_sort.detection import Detection
from yolo_counter import Yolo_detect
from deep_sort.tracker_danyow import Tracker
from collections import Counter
'''
yolo+deepsort+counting
'''
def run():

    video_path = '/Users/danyow/Desktop/yolov5_car_counter/carInHighway.mp4'
    cap = cv2.VideoCapture(video_path)
    ret,frame = cap.read()
    h,w = frame.shape[0:2]

    # yolo檢測模型
    model = Yolo_detect()
    # deepsort更新實例
    tracker = Tracker()
    # 計數變量
    count = 0
    # 計數字典
    count_dict = {}

    while ret:
        ret,frame = cap.read()
        
        
        '''观察'''
        detections = []# 用来卡尔曼滤波的列表
        # 返回yolo检测器的目标xy，
        res =  model.detect(frame)
        length = len(res[0]) # 多目标，所以依次获取
        for i in range(length) :
            detections.append(Detection(tlwh=res[0][i],confidence=res[1][i],feature=res[2][i]),)
                

        tracker.predict()
        tracker.update(detections)
        print('捕獲到軌跡',len(tracker.tracks)) 
        for track in tracker.tracks:
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = bbox
            
            track_id = track.track_id
            # count_dict[track_id] = (x1,y1)
            count_dict[track_id] = 0 
            if y1<h/2:
                count_dict[track_id] =  1 
            cv2.putText(frame,f'{track_id}' ,(int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (0, 0, 255), 1, cv2.LINE_AA)

        cv2.line(frame,(0,int(h/2)),(int(w),int(h/2)),color=(0,255,255))
        
        count += list(count_dict.values()).count(1)
        print(count)

        cv2.putText(frame,f'count = {count}' ,(50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (255, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('frame',frame)
        if cv2.waitKey(10)==27:
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ =='__main__':
    run()
    # deo_dict  ={0:1,1:1,2:0,30:0}
    # res = list(deo_dict.values()).count(1)
    # print(res)

    