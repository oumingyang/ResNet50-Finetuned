import _pickle as cPickle
import os

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

save_path = fzm_mosaic_path


def unpickle(file):
  with open(file, 'rb') as fo:
    dict = cPickle.load(fo)
  return dict

if __name__ == '__main__':

    print(unpickle(save_path + os.path.sep + 'data_batch_0'))