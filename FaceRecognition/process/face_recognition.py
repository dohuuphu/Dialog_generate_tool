import time
import torch
import utils
from process.identification import identification
from sklearn import preprocessing
from config.constant import EMBEDDING_DIMENSION, FRAMES_TO_PERSIST, MIN_SIZE_FOR_MOVEMENT, MOVEMENT_DETECTED_PERSISTENCE
import cv2
from datetime import datetime, timedelta


def face_recognition(url,model_info):
    # print(url)
    model_detection, model_recognition = model_info
    cap = cv2.VideoCapture(url) # Then start the webcam
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    delay_counter = 0
    movement_persistent_counter = 0
    count = 0
    # Init frame variables
    first_frame = None
    next_frame = None
    # LOOP!
    grid = []
    y1 = 0
    height_c = frame_height - 500
    width_c = frame_width - 400
    M = height_c//2
    N = width_c//2
    for y in range(0,height_c,M):
        for x in range(0, width_c, N):
            y1 = y + M
            x1 = x + N
            grid.append([x,y,x1,y1])
    count_part1 = 0
    count_part2 = 0
    count_part3 = 0
    count_part4 = 0
    flag= 0
    while True:
        # Set transient motion detected as false
        transient_movement_flag = False
        ret, frame = cap.read()
        while ret == False:
            print("Can't receive frame. Retrying ...")
            cap.release()
            cap = cv2.VideoCapture(url) # Then start the webcam
            ret, frame = cap.read()

        height , width , _ = frame.shape
        print(height,width)
        frame = frame[500:height,200:width-200]
        # Resize and save a greyscale version of the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur it to remove camera noise (reducing false positives)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # If the first frame is nothing, initialise it
        if first_frame is None: first_frame = gray    

        delay_counter += 1
        # Otherwise, set the first frame to compare as the previous frame
        # But only if the counter reaches the appriopriate value
        # The delay is to allow relatively slow motions to be counted as large
        # motions if they're spread out far enough
        if delay_counter > 1:
            delay_counter = 0
            first_frame = next_frame

            
        # Set the next frame to compare (the current frame)
        next_frame = gray

        # Compare the two frames, find the difference
        frame_delta = cv2.absdiff(first_frame, next_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Fill in holes via dilate(), and find contours of the thesholds
        thresh = cv2.dilate(thresh, None, iterations = 2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        for c in cnts:
        # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)
            
            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
                # transient_movement_flag = True
                count += 1
                part = 0
                for x_min, y_min, x_max, y_max in grid:
                    part += 1
                    if(x_min <= x and  x <= x_max and y_min<=y and y <= y_max):
                        if part == 1:
                            count_part1 +=1
                            #print('Motion detected at part {}'.format(part)) 
                        if part == 2:
                            count_part2 +=1
                            #print('Motion detected at part {}'.format(part))    
                        if part == 3:
                            count_part3 +=1
                            #print('Motion detected at part {}'.format(part))    
                        if part == 4:
                            count_part4 +=1
                            #print('Motion detected at part {}'.format(part))     
        
        if count_part1 >= 10:
            sub_frame = frame[grid[0][1]:grid[0][3],grid[0][0]:grid[0][2]]
            count_part1 = 0
            flag = 1
        if count_part2 >= 10:
            sub_frame = frame[grid[1][1]:grid[1][3],grid[1][0]:grid[1][2]]
            count_part2 = 0
            flag = 1
        if count_part3 >= 10:
            sub_frame = frame[grid[2][1]:grid[2][3],grid[2][0]:grid[2][2]]
            count_part3 =0
            flag = 1
        if count_part4 >= 10:
            sub_frame = frame[grid[3][1]:grid[3][3],grid[3][0]:grid[3][2]]
            count_part4 = 0
            flag = 1
        
        if flag == 1:
            flag = 0
            print('Get here')
            total_information = []
            list_person_info = identification(sub_frame, model_detection, model_recognition, threshold_detect=0.7, threshold_recog=0.6)
            for person_info in list_person_info:
                name = person_info[0]
                crop_face = person_info[4]
                # bounding_box = person_info[1]
                # landmarks = person_info[2]
                # distance = round(float(person_info[3]), 2)
                total_information.append([name,crop_face])
                cv2.imwrite('./libs/result/{}_{}.jpg'.format(name,str((datetime.now()).strftime("%H_%M_%S"))),crop_face)

        # # The moment something moves momentarily, reset the persistent
        # # movement timer.
        # if transient_movement_flag == True:
        #     movement_persistent_flag = True
        #     movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

        # # As long as there was a recent transient movement, say a movement
        # # was detected    
        # if movement_persistent_counter > 0:
        #     text = "Movement Detected " + str(movement_persistent_counter)
        #     movement_persistent_counter -= 1
        # else:
        #     text = "No Movement Detected"

       

