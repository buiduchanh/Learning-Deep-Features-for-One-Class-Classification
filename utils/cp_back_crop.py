import os
import numpy as np
import cv2
import glob
import math  
def calculateDistance(x1,y1,x2,y2):  
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    return dist 
def cut_center_image(image, flag_draw_line=False, minLineLength=None, margin_min=10, margin_max=10):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100], dtype = 'uint8')
    upper_yellow = np.array([30, 255, 255], dtype='uint8')
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_not(gray_image, mask_yw)
    _ ,thresh2 = cv2.threshold(mask_yw_image,230, 255,cv2.THRESH_BINARY_INV)
    mask_yw_image = cv2.bitwise_not(thresh2, mask_yw)
    mask_yw_image = cv2.GaussianBlur(mask_yw_image,(15,15),0)
    edges = cv2.Canny(mask_yw_image, 75, 150)
    
    if minLineLength == None:
        minLineLength = edges.shape[1] - 20
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength = minLineLength, maxLineGap=250)
    
    h,w = image.shape[:2]
    
    # print(len(lines))
    draw_lines = []
    z = np.array([])
    vertical_z = np.array([])
    horizontal_z = np.array([])
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if abs(x1 - x2) < 4 and x1 > margin_min and x2 < w - margin_min:
#                 vertical
            if flag_draw_line:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255,255), 3)
            if vertical_z.shape[0] == 0:
                vertical_z = np.array([[x1, y1]])
            else:
                a = np.array([[x1, y1]])
                vertical_z = np.concatenate((a, vertical_z))
        if abs(y1 - y2) < 4 and y1 > margin_min and y2 < h - margin_min:
#                 horizontal
            if flag_draw_line:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255,255), 3)
            if horizontal_z.shape[0] == 0:
                horizontal_z = np.array([[x1, y1]])
            else:
                a = np.array([[x1, y1]])
                horizontal_z = np.concatenate((a, horizontal_z))
    
    vertical_z_max = vertical_z.max(axis=0)
    # print(vertical_z_max)
    vertical_z_min = vertical_z.min(axis=0)
    # print(vertical_z_min)
    x_min = int(vertical_z_min[0])
    x_max = int(vertical_z_max[0])
    
    horizontal_z_max = horizontal_z.max(axis=0)
    # print(horizontal_z_max)
    horizontal_z_min = horizontal_z.min(axis=0)
    # print(horizontal_z_min)
    y_min = int(horizontal_z_min[1])
    y_max = int(horizontal_z_max[1])
    
    cut_top_bottom_image = image[y_min:y_max, x_min:x_max]
    
    x1, y1, x2, y2 = x_min, y_min, x_max, y_max
    
    return edges, cut_top_bottom_image, (x1, y1, x2, y2)
# https://stackoverflow.com/a/20679579
def line2point(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C
def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
    
# https://pysource.com/2018/03/07/lines-detection-with-hough-transform-opencv-3-4-with-python-3-tutorial-21/
def cut_top_bottom(image, flag_draw_line=False, minLineLength=None, margin_min=0, margin_max=0):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100], dtype = 'uint8')
    upper_yellow = np.array([30, 255, 255], dtype='uint8')
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_not(gray_image, mask_yw)
    _ ,thresh2 = cv2.threshold(mask_yw_image,230, 255,cv2.THRESH_BINARY_INV)
    mask_yw_image = cv2.bitwise_not(thresh2, mask_yw)
    blur = cv2.GaussianBlur(mask_yw_image,(15,15),0)
    edges = cv2.Canny(blur, 75, 150)
    
    if minLineLength == None:
        minLineLength = edges.shape[1] - 20
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength = minLineLength, maxLineGap=250)
    
    # print(len(lines))
    draw_lines = []
    horizontal_lines = []
    vertical_lines = []
    z = np.array([])
    vertical_z = np.array([])
    horizontal_z = np.array([])
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        
        if abs(x1 - x2) < 20:
            if flag_draw_line:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255,255), 3)
            vertical_lines.append(line[0])
            if vertical_z.shape[0] == 0:
                vertical_z = np.array([[x1, y1]])
            else:
                a = np.array([[x1, y1]])
                vertical_z = np.concatenate((a, vertical_z))
            
        if abs(y1 - y2) < 20:
            if flag_draw_line:
                cv2.line(image, (x1, y1), (x2, y2), (0, 120,255), 3)
            horizontal_lines.append(line[0])
            if horizontal_z.shape[0] == 0:
                horizontal_z = np.array([[x1, y1]])
            else:
                a = np.array([[x1, y1]])
                horizontal_z = np.concatenate((a, horizontal_z))
                
    vertical_z_max = vertical_z.max(axis=0)
    # print(vertical_z_max)
    vertical_z_min = vertical_z.min(axis=0)
    # print(vertical_z_min)
    x_min = int(vertical_z_min[0]) + margin_min
    x_max = int(vertical_z_max[0]) - margin_max
    
    horizontal_z_max = horizontal_z.max(axis=0)
    # print(horizontal_z_max)
    horizontal_z_min = horizontal_z.min(axis=0)
    # print(horizontal_z_min)
    y_min = int(horizontal_z_min[1]) + margin_min
    y_max = int(horizontal_z_max[1]) - margin_max
    
    h,w = image.shape[:2]
    
    cut_top_bottom_image = image[y_min:y_max, x_min:x_max]
    
    x1, y1, x2, y2 = x_min, y_min, x_max, y_max
#     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
#     cv2.rectangle(image, (x_max, y_max), (h, w), (255,0,0), 2)
    
    return cut_top_bottom_image, (x1, y1, x2, y2)
def crop_save_image_back(image):
    cut_top_bottom_image, rec_1 = cut_top_bottom(image, flag_draw_line=False, minLineLength=image.shape[0]/2)
    x1 = rec_1[0]
    y1 = rec_1[1]
    x2 = rec_1[2]
    y2 = rec_1[3]
    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
    edge_image = cut_top_bottom_image.copy()
    _ , edges, rec_2 = cut_center_image(edge_image, flag_draw_line=False, minLineLength=image.shape[0]/4)
    op2_image = edge_image.copy()
    h,w = op2_image.shape[:2]
    x1 = int(rec_2[0] - rec_2[0]/2)
    y1 = int(rec_2[1] - rec_2[1]/2)
    x2 = int(rec_2[2] + (w - rec_2[2])/2)
    y2 = int(rec_2[3] + (h - rec_2[3])/2)
    # cv2.rectangle(op2_image, (x1, y1), (x2, y2), (0, 255,120), 2)
    # save_image = op2_image[y1:y2, x1:x2]
    # return save_image
    x1 = int(rec_1[0] + x1)
    y1 = int(rec_1[1] + y1)
    x2 = int(rec_1[2] - (w - x2))
    y2 = int(rec_1[3] - (h - y2))
    # return save_image, x1, y1, x2, y2
    return x1, y1, x2, y2