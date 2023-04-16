import numpy as np 
import cv2 
import pandas as pd 


cap = cv2.VideoCapture('carInHighway.mov')
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

width = int(width)
height = int(height)
print(frames_count,fps,width,height)


# 创建pd数列来进行跟踪
df = pd.DataFrame(index=range(int(frames_count)))
df.index.name ='Frames'
framenumber = 0  # keeps track of current frame
carscrossedup = 0  # keeps track of cars that crossed up
carscrosseddown = 0  # keeps track of cars that crossed down
carids = []  # blank list to add car ids
caridscrossed = []  # blank list to add car ids that have crossed
totalcars = 0  # keeps track of total cars

# 创建背景剪法器
bg_mask = cv2.createBackgroundSubtractorMOG2()

ret,frame = cap.read()
ratio = 0.5 # 重新设置速率
image = cv2.resize(frame,(0,0),None,ratio,ratio)
width2 ,height2,channels = image.shape 

while True:
    ret,frame = cap.read()
    if ret:
        image = cv2.resize(frame,(0,0),None,ratio,ratio)
        # 对数据图片进行预处理
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        fgmask = bg_mask.apply(gray)

        # apply different thresholds to fgmask to try and isolate cars 
        # just have to keep playing around with settings until cars are easily identifiable 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
        closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel)
        retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
        # 创建边框
        # im2, contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # use convex hull to create polygon around contours 
        hull = [cv2.convexHull(c) for c in contours]
        # draw contours 
        cv2.drawContours(image,hull,-1,(0,255,0),3)
        # line created to stop counting contours,needed as cars in distance become one big contour 
        linepos = 225 
        cv2.line(image,(0,linepos),(width,linepos),(255,0,0),5)

        #line y position created to count contours 
        linepos2 = 250 
        cv2.line(image,(0,linepos2),(width,linepos2),(255,0,0),5)

        # min area for contours in case a bunch of small noise contours are created 
        minarea = 300 
        # max area for contours,can be quite large for buses
        maxarea = 50000

        # vector for the x and y locations of contour centroids in current frame 
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))

        # cycle through all contours in current frame  
        for i in range(len(contours)):
            if hierarchy[0,i,3]==-1:# using hierarchy to only count parent contours (contours not within others)
                area = cv2.contourArea(contours[-1])
                if minarea <area<maxarea: # area threshold for contour 
                    # 计算中心点
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    if cy>linepos:
                        # filters out contours that are above line 
                        x,y,w,h = cv2.boudingRect(cnt)
                        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    .3, (0, 0, 255), 1)
 
                        cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                       line_type=cv2.LINE_AA)
 
                        cxx[i] = cx 
                        cyy[i] = cy
        cxx = cxx[cxx!=0]
        cyy = cyy[cyy!=0]
        '''keep track of Centroids'''
        minx_index2 = []
        miny_index2 = []
        # maximum allowable radius for current frame centroid to be considered the same centroid from previous frame 
        maxrad = 25 

        # the section below keeps track of the centroid and assigns them to old carids or new carids 
        if len(cxx):
            if not carids:# if carids is empty
                for i in range(len(cxx)):
                    
                    carids.append(i) # adds a car id to the empty list carids 

                    df[str[carids[i]]] = '' # adds a column to the dataframe corresponding to a carid 
                    # assigns the centroid values to the current frame(row) and carid(column)
                    df.at[int(framenumber),str(carids[i])] = [cxx[i],cyy[i]]
                    totalcars  = carids[i]+1 
            else:
                dx = np.zeros((len(cxx),len(carids)))
                dy = np.zeros((len(cyy),len(carids)))
                '''
                # loops through all centroids
                '''
                for i in range(len(cxx)):
                    for j in range(len(carids)):
                        oldcxcy = df.iloc[int(framenumber-1)][str(carids[j])]
                        curcxcy = np.array([cxx[i],cyy[i]])

                        if not oldcxcy:
                            continue
                        else:
                            dx[i,j] = oldcxcy[0] - curcxcy[0]
                            dy[i,j] = oldcxcy[1]-oldcxcy[1]
                '''
                 # loops through all current car ids
                 '''
                for j in range(len(carids)):
                    sumsum = np.abs(dx[:,j]) + np.abs(dy[:,j])

                    # find which index carid had the min difference and this is true index
                    correctindextrue = np.argmin(np.abs(sumsum))
                    minx_index = correctindextrue
                    miny_index = correctindextrue

                    # acquires delta values of the minium delete in order to check if it is within radius later on 
                    mindx = dx[minx_index,j]
                    mindy = dx[miny_index,j]

                    if mindx == 0 and mindy == 0 and np.all(dx[:,j]==0) and np.all(dy[:,j]==0):
                        # check if minimum value is 0 and checks if all deltas are zero since this is empty set 
                        continue
                    else:
                        if np.abs(mindx)<maxrad and np.abs(mindy)<maxrad:
                            df.at[int(framenumber),str(carids[j])] = [cxx[minx_index],cyy[miny_index]]
                            minx_index2.append(minx_index)
                            miny_index2.append(miny_index)
                '''
                # loops through all centroids
                '''
                for i in range(len(cxx)):
                    if i not in minx_index2 and miny_index2:
                        df[str[totalcars]]=''
                        totalcars = totalcars +1 
                        t = totalcars -1 
                        carids.append(t)
                        df.at[int(framenumber),str(t)]=[cxx[i],cyy[i]]
                    elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                        df[str[totalcars]]=''
                        totalcars = totalcars+1 
                        t = totalcars -1 
                        carids.append(t)
                        df.at[int(framenumber),str(t)]=[cxx[i],cyy[i]]


        '''
        Counting cars 
        '''
        currentcars = 0 
        currentcarsindex = []
        for i in range(len(carids)):
            if df.at[int(framenumber),str(carids[i])]!='':
                # checks the current frame to see which car ids are activate 
                currentcars = currentcars +1 
                currentcarsindex.append(i)
        for i in range(len(currentcars)):
            curcent = df.iloc[int(framenumber)][str(carids[currentcarsindex[i]])]
            oldcent = df.iloc[int(framenumber-1)][str(carids[currentcarsindex[i][i]])]

            if curcent:
                cv2.putText(image, "Centroid" + str(curcent[0]) + "," + str(curcent[1]),
                            (int(curcent[0]), int(curcent[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)
 
                cv2.putText(image, "ID:" + str(carids[currentcarsindex[i]]), (int(curcent[0]), int(curcent[1] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)
 
                cv2.drawMarker(image, (int(curcent[0]), int(curcent[1])), (0, 0, 255), cv2.MARKER_STAR, markerSize=5,
                               thickness=1, line_type=cv2.LINE_AA)
                
                if oldcent:
                    xstart = oldcent[0] - maxrad
                    ystart = oldcent[1] - maxrad 

                    xwidth = oldcent[0] + maxrad 
                    yheight = oldcent[1] + maxrad
                    cv2.rectangle(image,(int(xstart),int(ystart)),(int(xwidth),int(yheight)),(0,125,0),1)

                    # checks if old centroid is on or below line and current is on or above line to count cars and that car hasn't been counted yet
                    if oldcent[1] >= linepos2 and curcent[1] <= linepos2 and carids[
                        currentcarsindex[i]] not in caridscrossed:
 
                        carscrossedup = carscrossedup + 1
                        cv2.line(image, (0, linepos2), (width, linepos2), (0, 0, 255), 5)
                        caridscrossed.append(
                            currentcarsindex[i])  # adds car id to list of count cars to prevent double counting
 
                    # checks if old centroid is on or above line and curcent is on or below line
                    # to count cars and that car hasn't been counted yet
                    elif oldcent[1] <= linepos2 and curcent[1] >= linepos2 and carids[
                        currentcarsindex[i]] not in caridscrossed:
 
                        carscrosseddown = carscrosseddown + 1
                        cv2.line(image, (0, linepos2), (width, linepos2), (0, 0, 125), 5)
                        caridscrossed.append(currentcarsindex[i])


        cv2.rectangle(image, (0, 0), (250, 100), (255, 0, 0), -1)  # background rectangle for on-screen text
 
        cv2.putText(image, "Cars in Area: " + str(currentcars), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
 
        cv2.putText(image, "Cars Crossed Up: " + str(carscrossedup), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0),
                    1)
 
        cv2.putText(image, "Cars Crossed Down: " + str(carscrosseddown), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (0, 170, 0), 1)
 
        cv2.putText(image, "Total Cars Detected: " + str(len(carids)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (0, 170, 0), 1)
 
        cv2.putText(image, "Frame: " + str(framenumber) + ' of ' + str(frames_count), (0, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 170, 0), 1)
 
        cv2.putText(image, 'Time: ' + str(round(framenumber / fps, 2)) + ' sec of ' + str(round(frames_count / fps, 2))
                    + ' sec', (0, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)
 
        # displays images and transformations
        cv2.imshow("countours", image)
        cv2.moveWindow("countours", 0, 0)
 
        cv2.imshow("fgmask", fgmask)
        cv2.moveWindow("fgmask", int(width * ratio), 0)
 
        cv2.imshow("closing", closing)
        cv2.moveWindow("closing", width, 0)
 
        cv2.imshow("opening", opening)
        cv2.moveWindow("opening", 0, int(height * ratio))
 
        cv2.imshow("dilation", dilation)
        cv2.moveWindow("dilation", int(width * ratio), int(height * ratio))
 
        cv2.imshow("binary", bins)
        cv2.moveWindow("binary", width, int(height * ratio))
        
        framenumber += 1 
        if cv2.waitKey(10)==27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
df.to_csv('traffic.csv', sep=',')

