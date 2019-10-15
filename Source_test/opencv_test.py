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
        #OpenCV import color of picture by BGR, need convert to RGB.
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # im = np.transpose(im, [2, 1, 0])
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
    if os.path.isdir(filename):
        print ("Can't found data file!")
    else:
        f = open(filename)
        lines = f.read().splitlines()
        print(lines)
        count = len(lines)
        data = np.zeros((count, DATA_LEN), dtype=np.uint8)
        #label = np.zeros(count, dtype=np.uint8)
        lst = [ln.split(' ')[0] for ln in lines]
        #print("file list:",lst)
        label = [int(ln.split(' ')[1]) for ln in lines]
        #print("label list:",label)
    
    idx = 0
    s, c = SHAPE, CHANNEL_LEN
    for ln in lines:
        fname, lab = ln.split(' ')
        im = imread(os.path.join(data_path, fname), shape=s, color='RGB')
        '''
        im = cv2.imread(os.path.join(data_path, fname), cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (s, s))
        '''
        data[idx,:c] =  np.reshape(im[:,:,0], c)
        data[idx, c:2*c] = np.reshape(im[:,:,1], c)
        data[idx, 2*c:] = np.reshape(im[:,:,2], c)
        label[idx] = int(lab)
        idx = idx + 1
      
    return data, label, lst

data_path = fzm_mosaic_path
file_list = fzm_mosaic_path + os.path.sep + "images.list"
save_path = fzm_mosaic_path
read_data(file_list, data_path)