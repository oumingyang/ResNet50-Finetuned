import cv2
import numpy as np

import os

from keras.preprocessing import image

from keras.applications.resnet50 import preprocess_input, decode_predictions


#Get current file path
current_path = os.path.abspath(__file__)

#Get current file father_dir
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
source_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Source_Image")
mosaic_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Mask_Image")

#Path of image for train
fzm_image_path = os.path.join(os.path.abspath(source_image_path + os.path.sep),"fzm")
fzm_img = os.path.join(os.path.abspath(fzm_image_path + os.path.sep),"fangzuming0.jpg")

#Path of image done mosaic
fzm_mosaic_path = os.path.join(os.path.abspath(mosaic_image_path + os.path.sep),"fzm")
fzm_mosaic = os.path.join(os.path.abspath(fzm_mosaic_path + os.path.sep),"fzm0_mosaic.jpg")

img = image.load_img(fzm_img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)

y = np.array([[1,0]] + [[0,1]])


# cv2.namedWindow("src")
# cv2.imshow("src",src)
# key = cv2.waitKey()

