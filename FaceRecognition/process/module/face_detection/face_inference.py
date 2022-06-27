import numpy as np
import os
import cv2 as cv

from FaceRecognition.process.module.face_detection.utils.alignment import get_reference_facial_points, warp_and_crop_face
c_ = 0

def create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)



def process_img(img, facial_5_points, output_size):
    global c_
    
    # test_img = draw_landmarks(img, facial_5_points)
    facial_points = np.array(facial_5_points)

    default_square = True
    inner_padding_factor = 0.5
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    dst_img, ref_point = warp_and_crop_face(img, facial_points, reference_pts=reference_5pts, crop_size=output_size)

    #draw landmark
    test_img = img.copy()
    test_img = draw_landmarks(test_img, facial_5_points)


    is_good_face = is_frontFaceing(facial_5_points)
    t = 'save' if is_good_face  else 'discard'


    c_+=1
    if is_good_face is False:
        cv.putText(img=test_img, text=t, org=(15, 15), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0),thickness=1)

        cv.imwrite(f'/mnt/c/Users/phudh/Desktop/src/dialog_system/draf_2/{c_}.jpg',test_img )
    

    if is_good_face: 
        top, bottom, left, right = mouth_ROI(ref_point)
    
        return dst_img, dst_img[top:bottom, left:right]
    else: 
        return  None, None

    

def get_face_area(img, detector, threshold, scales = [640, 1200], crop_img = True, img_raw = None):
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    
    im_scale = float(target_size) / float(im_size_min)
    
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False

    faces, landmarks = detector.detect(img,
                                    threshold,
                                    scales=scales,
                                    do_flip=flip)

    result_faces = []
    result_mouth = []
    result_landmark = []
    if crop_img:
        for facial_5_points in landmarks:
            crop_face_img, mouth = process_img(img, facial_5_points, (112,112))
            if crop_face_img is not None:
                result_mouth.append(mouth)
                result_faces.append(crop_face_img)
    else:
        if len(landmarks)>0 and is_frontFaceing(landmarks[0]) :
            result_faces = [img_raw]
            for landmark in landmarks:
                result_landmark.append(landmark)
                top, bottom, left, right = mouth_ROI(landmark)
                result_mouth.append(img_raw[top:bottom, left:right])

            


    return result_faces, result_mouth, result_landmark


def draw_landmarks(img, landmark):
    img_ = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    for point in landmark:
        cv.circle(img, (int(point[0]), int(point[1])), 3, (0,0,255), -1)

    return img
    
def getName_landmark(landmark): # [0]:W [1]:H
    landmark_revert = [list(reversed(i)) for i in landmark.astype(np.int16).tolist()]
    temp = sorted(landmark_revert)
    top1, top2, nose, bottom1, bottom2 = [list(reversed(i)) for i in temp]


    if bottom1[0] < bottom2[0]:
        m_left = bottom1
        m_right = bottom2
    else:
        m_left = bottom2
        m_right = bottom1

    if top1[0] < top2[0]:
        e_left = top1
        e_right = top2
    else:
        e_left = top2
        e_right = top1
    
    return e_left, e_right, nose, m_left, m_right

def mouth_ROI(landmark):
    try:
        e_left, e_right, nose, m_left, m_right =  getName_landmark(landmark)

        #create box: [0]:W [1]:H
        delta = 5

        top = m_left[1] - 2*delta
        bottom = m_left[1] + 2*delta

        left = m_left[0] - delta
        right = m_right[0] + delta

        return top, bottom, left, right

    except:
        pass

# def is_frontFaceing(img_landmark):
#     try:
#         (e_left, e_right, nose, m_left, m_right), area =  find_landmark(img_landmark)

#         if abs(e_left[0] - e_right[0]) < 10 or abs(m_left[0] - m_right[0]) < 10:
#             return False, 1000
#         else:
#             return True, area
#     except: 
#         return False, 1000

def is_frontFaceing(landmark):
    try:
        e_left, e_right, nose, m_left, m_right =  getName_landmark(landmark)

        if abs(e_left[0] - e_right[0]) < 10 or abs(m_left[0] - m_right[0]) < 10:
            return False
        else:
            return True
    except: 
        return False

def find_landmark(img_landmark):
    landmark = []
    # find contour
    _, _, red = cv.split(img_landmark)
    ret, thresh = cv.threshold(red, 240, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # compute the center of the contour
    for cnt in contours:
        M = cv.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        landmark.append([cX, cY]) #width, Height

        area = cv.contourArea(cnt) # get area
    

    return landmark, area