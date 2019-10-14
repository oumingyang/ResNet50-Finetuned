import cv2
import os
import numpy as np


#Get current file path
current_path = os.path.abspath(__file__)
 #Get current file father_dir
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
source_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Source_image")
mosaic_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Mosaic_image")

#Path of image for train
fzm_image_path = os.path.join(os.path.abspath(source_image_path + os.path.sep),"fzm")
cl_image_path = os.path.join(os.path.abspath(source_image_path + os.path.sep),"cl")
fzm_img = os.path.join(os.path.abspath(fzm_image_path + os.path.sep),"fangzuming0.jpg")
fzm_mosaic_path = os.path.join(os.path.abspath(mosaic_image_path + os.path.sep),"fzm")
fzm_mosaic = os.path.join(os.path.abspath(fzm_mosaic_path + os.path.sep),"fzm0_mosaic.jpg")


DATA_LEN = 3072
CHANNEL_LEN = 1024
SHAPE = 32

def imread(im_path, shape=None, color="RGB", mode=cv2.IMREAD_UNCHANGED):
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    im_row=im[0]
  
    print("im_shape:",im_row)
    print(im_row.shape)

    im_clo = im[:,0]
    print(im_clo.shape)
    print("im_col_shape:", im_clo)

    if color == "RGB":
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # im = np.transpose(im, [2, 1, 0])
    if shape != None:
        assert isinstance(shape, int) 
        im = cv2.resize(im, (shape, shape))
    return im



imread(fzm_img)