import cv2
import os
import numpy as np
import json

def imread(im_path, shape=None, color="RGB", mode=cv2.IMREAD_UNCHANGED):
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    im_row=im[0]

    im_clo = im[:,0]

    if color == "RGB":
        #OpenCV import color of picture by BGR, need convert to RGB.
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if shape != None:
        assert isinstance(shape, int) 
        im = cv2.resize(im, (shape, shape))
    return im

def read_data(filename, data_path, shape=None, color='RGB'):
    """
    filename (str): a file 
    data file is stored in such format:
    image_name  label
    data_path (str): image data folder
    return (numpy): a array of image and a array of label
    """ 
    DATA_LEN = 3072
    CHANNEL_LEN = 1024
    SHAPE = 32

    if os.path.isdir(filename):
        print ("Can't found data file!")
    else:
        # read Json data
        with open(filename, "r", encoding="utf-8") as fr:
            json_data = [json.loads(line) for line in fr]

        s, c = SHAPE, CHANNEL_LEN
        file_list = []
        label_list = []

        for i in range(len(json_data)):

            category = json_data[i]['category']
            label = json_data[i]["label"]
            filename = json_data[i]["file_name"]
            count = len(filename)
            image_data_path = os.path.join(data_path, category)
            print(image_data_path)
            for i in range(count):
                category_file = category + "_" + filename[i]
                file_list.append(category_file) 
                data = np.zeros((count, DATA_LEN), dtype = np.uint8)
                im = imread(os.path.join(image_data_path, filename[i]), shape=s, color='RGB')
                data[i,:c] =  np.reshape(im[:,:,0], c)
                data[i, c:2*c] = np.reshape(im[:,:,1], c)
                data[i, 2*c:] = np.reshape(im[:,:,2], c)
                label_list.append(label)

    return data, label_list, file_list

if __name__ == "__main__":

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

    # Read_data
    read_data(image_train_record_file, image_train_path)