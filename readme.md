# yolov5 + deepsort = object tracking (object counting)

## basic:
yolov5 for car detection
deepsort for car tracking 

## features:
if the car `y` coordinate greater than the half of the frame height ,then we add 1 to the total count 

## main code:
- main_count.py:
    yolo+deepsort+counting
- main.py
    yolo+deepsort
- yolo_counter.py
    > You can specify the required category to identify
    yolo detection through opencv dnn module 


## output demo:
🚗output.mp4

## ref
1. https://github.com/ultralytics/yolov5

2. https://github.com/nwojke/deep_sort

