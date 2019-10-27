from keras.models import Model, load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from matplotlib import pyplot as plt

import os


def get_file_path(root_path, dir_list, file_list):
    # get all of the dir and file fellow root_path
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # get dir or file path
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            #Recursive
            get_file_path(dir_file_path, dir_list, file_list)
        else:
            file_list.append(dir_file_path)


if __name__ == "__main__":
    
    #Get current file path
    current_path = os.path.abspath(__file__)

    #Get current file father_dirs_path
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")    
    mosaic_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "../.."),"Mosaic_image/fzm")

    root_path = mosaic_image_path

    dir_list = []
    file_list = []
    
    get_file_path(root_path, dir_list, file_list)

    module = load_model('resnet50_final.h5')

    for file in file_list:
    #load moudle
        
        img = image.load_img(file, target_size=(224,224))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        preds = module.predict(x, batch_size=1)
        if preds[:,[0]] < preds[:,[1]]:
            print("Mosaic working for file :\n", file, "\n")
            print("Prediction of file:", preds, "\n")
    
    # print("Prediction of file:", file, "is:\n", preds, "\n")
    # #decode the result into a list of tuples(class, description, probability)
    # # print('Predicted:', decode_predictions(preds, top=2)[0])

    