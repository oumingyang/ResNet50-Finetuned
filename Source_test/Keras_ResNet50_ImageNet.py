from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from matplotlib import pyplot as plt

import os
#Get current file path
current_path = os.path.abspath(__file__)

#Get current file father_dirs_path
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")    
original_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Source_Image")
mask_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Mask_Image")

#Orignal_Image_path
image_path = os.path.join(os.path.abspath(original_image_path + os.path.sep),"human.jpg")

#load moudle
module = ResNet50(weights = "imagenet")

img_path = image_path
img = image.load_img(img_path, target_size=(224,224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)

preds = module.predict(x)

#decode the result into a list of tuples(class, description, probability)
print('Predicted:', decode_predictions(preds, top=3)[0])
