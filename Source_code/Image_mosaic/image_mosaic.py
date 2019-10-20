#coding:utf-8
'''
马赛克效果
'''
import cv2
import numpy as np

import os

#Get current file path
current_path = os.path.abspath(__file__)

#Get current file father_dir
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
source_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "../.."),"Source_image")
mosaic_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "../.."),"Mosaic_image")

#Path of image for train
fzm_image_path = os.path.join(os.path.abspath(source_image_path + os.path.sep),"fzm")
fzm_img = os.path.join(os.path.abspath(fzm_image_path + os.path.sep),"fangzuming0.jpg")

#Path of image done mosaic
fzm_mosaic_path = os.path.join(os.path.abspath(mosaic_image_path + os.path.sep),"fzm")


def mosaic(selected_image,nsize=6):
    rows,cols,_ = selected_image.shape
    dist = selected_image.copy()
    # 划分小方块，每个小方块填充随机颜色
    for y in range(0,rows,nsize):
        for x in range(0,cols,nsize):
            dist[y:y+nsize,x:x+nsize] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
    return dist

"""
step_size = lambada
detecting_density_in_y = y_fps
detecting_density_in_x = x_fps
step : point_start : end_point
1 : 0 : lambada-1
2 : lambada : 2lambada-1
...
n : (n-1)lambada : n*lambada-1

set rows_resize = x_fps*lambada_x-1
    cols_resize = y_fps*lambada_y-1 
"""

point_start = {'x':0,'y':0}
point_end = {'x':0,'y':0}

rows = 0
cols = 0

y_height = 18
x_width = 18

src = cv2.imread(fzm_img)
rows,cols,_ = src.shape


# 处理选择的矩形
rect = {}
rect['y'] = point_start['y']
rect['x'] = point_start['x']

rect['width'] = x_width
rect['height'] = y_height

# select_window

for rect['y'] in range(0,rows,y_height):
    for rect['x'] in range(0,cols,x_width):
        src_copy = src.copy()
        select_image = src_copy[rect['y']:rect['y']+rect['height'],
            rect['x']:rect['x']+rect['width']]
        result = mosaic(select_image)
        # 将处理完成的区域合并回原图像
        src_copy[rect['y']:rect['y']+rect['height'],
            rect['x']:rect['x']+rect['width']] = cv2.addWeighted(result, 1.0, select_image, 0.0, 0.0)
        suffix = "fzm_mosaic" + str(rect['y']) + "_" + str(rect['x']) + ".jpg" 
        fzm_mosaic = os.path.join(os.path.abspath(fzm_mosaic_path + os.path.sep), suffix)
        cv2.imwrite(fzm_mosaic,src_copy)

        # cv2.imshow('result',src_copy)
        # cv2.waitKey()
        # cv2.destroyAllWindows()