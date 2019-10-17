import os, cv2
from pickled import *
from dataset_tolist import *


if __name__ == '__main__':
      
  #Get current file path
  current_path = os.path.abspath(__file__)
  #Get current file father_dir
  father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
  source_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "../.."),"Source_image")
  mosaic_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "../.."),"Mosaic_image")
  dataset_train_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "../.."),"Data_train")
  dataset_valiate_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "../.."),"Data_valiate")

  #Path of data and datasets

  image_train_path = source_image_path
  image_train_record_file = source_image_path + os.path.sep + "data.json"
  dataset_train_save_path = dataset_train_path
  dataset_valiate_save_path = dataset_valiate_path


  # pickled train_dataset
  data, label, file_name_list = read_data(image_train_record_file, image_train_path, shape=32)
  pickled(dataset_train_save_path, data, label, file_name_list, bin_num = 1, mode="train")