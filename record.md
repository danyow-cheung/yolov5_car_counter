# 目标

kaggle的比赛

https://www.kaggle.com/competitions

----

因为第一次合作，要<u>磨合</u>合作模式



## 小试牛刀

**topic**:yolo的计数实现

目标结果

<img src = 'https://miro.medium.com/v2/resize:fit:1400/1*DRr0eqS8e8fzT33ZAhXPQA.gif'>



已有结果

opencv使用yolov5来识别车辆,模型已经变成.onnx,

https://github.com/danyow-cheung/yolov5_car_counter



我没解决的问题：

首先要**跟踪目标**，再实现计数功能。



### Steps

> 7.25-7.29

1. 读源码，关于counter的源码repo有，理解先。然后两天后（7.27）讲一下思路

   - https://github.com/dyh/unbox_yolov5_deepsort_counting

   - https://jorgestutorials.com/pycvtraffic.html @HHHangO

     

2. 讨论，其他的实现方法。

3. 各自去实现3天，给出结果。尽可能复习目标结果。

   

---

如果合作的不错，就继续做kaggle比赛

否则的话，写篇blog来介绍整个工作流程。





### 2023.7.27

1. 還是選擇yolo+deepsort的方法，先把這一步做了。（7.29完成）

   >  https://github.com/computervisioneng/object-tracking-yolov8-deep-sort 直接手把手教你怎麼寫了。實現的方法很多，看最後結果


   **20230731 更新：發現新的問題**
   目前已經實現目標跟蹤，剩下來的是記錄id計數，這一步又要怎麼實現？
   **20230801**
   大致完成計數和跟蹤，但是有個問題發現
   1. deepsort自己檢測的話，也會錯誤。會跟蹤錯誤
   2. 計數這套邏輯，會出現負數 😆
   3. 其次就是yolov5 自己的問題檢測不夠即時咯
   4. 

2. 找測試視頻，驗證這套方法行不行。總結不行的原因（7.30）

3. 

   





