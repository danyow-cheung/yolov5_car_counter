import os
import cv2
import numpy as np
from deep_sort import nn_matching
from deep_sort.detection import Detection
from yolo_counter import Yolo_detect
from deep_sort.tracker_danyow import Tracker

colors = [0,255,255]
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
        bboxes = []
        # 返回yolo检测器的目标xy，
        res =  model.detect(frame)
        length = len(res[0]) # 多目标，所以依次获取
        for i in range(length) :
            bboxes.append(res[0][i])
            detections.append(Detection(tlwh=res[0][i],confidence=res[1][i],feature=res[2][i]),)
        

        tracker.predict()
        tracker.update(detections)
        # 可能是這一塊
        # for track,box in zip(tracker.tracks,bboxes):
        for track in tracker.tracks:
            bbox = track.to_tlbr()

            x1, y1, x2, y2 = bbox
            # 現在是拿到了id但是感覺對不上的？
            track_id = track.track_id

            cv2.putText(frame,f'{track_id}' ,(int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (0, 0, 255), 1, cv2.LINE_AA)

            
        cv2.imshow('frame',frame)
        if cv2.waitKey(10)==27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ =='__main__':
    run()
    # tracker = Tracker()
