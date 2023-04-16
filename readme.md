# YOLO5 +  Counter
> 先把yolov5s.pt 转换为 .onxx 格式
 

~~https://www.youtube.com/watch?v=WgPbbWmnXJ8&t=5031s~~ 太多包，不可

使用像素值的差异进行计数，应该不难。

错误解决：https://forum.opencv.org/t/error-when-reading-yolo5-as-onnx-using-cv2/11507/3

**逐帧读入，但是在运动过程如何做到不重复计数？**

还是要使用ID索引进行跟踪每辆车的轨迹，使用ID索引要确保每辆车的ID不会改变。

更新，根据iD的索引的话，随着视频帧的移动，ID出现改变

## 2023.4.16 learning 
> 1. https://github.com/charnkanit/Yolov5-Vehicle-Counting/blob/main/track.py
> 2. https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/
> 3. https://jorgestutorials.com/pycvtraffic.html

