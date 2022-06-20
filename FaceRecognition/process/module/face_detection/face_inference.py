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
    #draw landmark
    img_landmark = draw_landmarks(img, facial_5_points)

    facial_points = np.array(facial_5_points)

    default_square = True
    inner_padding_factor = 0.5
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    dst_img = warp_and_crop_face(img, facial_points, reference_pts=reference_5pts, crop_size=output_size)

    cropImg_landmark = warp_and_crop_face(img_landmark, facial_points, reference_pts=reference_5pts, crop_size=output_size)

    is_good_face, area_point = is_frontFaceing(cropImg_landmark)
    t = 'save' if is_good_face  else 'discard'
    # if area_point< 20 :
    #     dst_img = cv.blur(dst_img, (3, 3))
    #     cropImg_landmark = cv.blur(cropImg_landmark, (3, 3))


    c_+=1
    cv.putText(img=cropImg_landmark, text=t, org=(15, 15), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0),thickness=1)

    cv.imwrite(f'/mnt/c/Users/phudh/Desktop/src/dialog_system/draf_2/{c_}.jpg',cropImg_landmark )
    

    if is_good_face: 
        return dst_img
    else: 
        return    

    

def get_face_area(img, detector, threshold, scales = [640, 1200]):
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

    crop_faces = []
    for facial_5_points in landmarks:
        crop_face_img = process_img(img, facial_5_points, (112,112))
        if crop_face_img is not None:
            crop_faces.append(crop_face_img)

    return crop_faces, faces, landmarks


def draw_landmarks(img, landmark):
    
    img_ = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    for point in landmark:
        cv.circle(img_, (int(point[0]), int(point[1])), 5, (0,0,255), -1)

    return img_
    
    
def is_frontFaceing(img_landmark):
    try:
        (e_left, e_right, nose, m_left, m_right), area =  find_landmark(img_landmark)

        if abs(e_left[0] - e_right[0]) < 10 or abs(m_left[0] - m_right[0]) < 10:
            return False, 1000
        else:
            return True, area
    except: 
        return False, 1000

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