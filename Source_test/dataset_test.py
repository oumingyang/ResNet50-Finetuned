import os, cv2
from pickled import *
from image_read_test import *

#Get current file path
current_path = os.path.abspath(__file__)
 #Get current file father_dir
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
source_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Source_Image")
mosaic_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Mask_Image")

#Path of image for train
fzm_image_path = os.path.join(os.path.abspath(source_image_path + os.path.sep),"fzm")
cl_image_path = os.path.join(os.path.abspath(source_image_path + os.path.sep),"cl")
fzm_img = os.path.join(os.path.abspath(fzm_image_path + os.path.sep),"fangzuming0.jpg")
fzm_mosaic_path = os.path.join(os.path.abspath(mosaic_image_path + os.path.sep),"fzm")
fzm_mosaic = os.path.join(os.path.abspath(fzm_mosaic_path + os.path.sep),"fzm0_mosaic.jpg")


data_path = fzm_mosaic_path
file_list = fzm_mosaic_path + os.path.sep + "images.list"
save_path = fzm_mosaic_path

if __name__ == '__main__':
  data, label, lst = read_data(file_list, data_path, shape=32)
  pickled(save_path, data, label, lst, bin_num = 1)