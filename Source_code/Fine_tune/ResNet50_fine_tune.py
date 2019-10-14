import math, json, os, sys
import numpy as np

import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, Flatten
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from PIL import Image
from keras import backend as K


#Get current file path
current_path = os.path.abspath(__file__)

#Get current file father_dirs_path
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")    
source_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Source_Image")
mask_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Mask_Image")

#Path of image for train
fzm_image_path = os.path.join(os.path.abspath(source_image_path + os.path.sep),"fzm")
cl_image_path = os.path.join(os.path.abspath(source_image_path + os.path.sep),"cl")

#Image_path
fzm_path = os.path.join(os.path.abspath(fzm_image_path + os.path.sep),"fzm")
cl_path = os.path.join(os.path.abspath(cl_image_path + os.path.sep),"cl")

# Data_dir = '/home/alan/work/vs_code/WorkSpace/Image_mask/Source_Code/data'
# Train_dir = os.path.join(Data_dir, "train")
# Valid_dir = os.path.join(Data_dir, "valid")
# Size = (244,244)
# Batch_size = 16

seed = 42
epochs = 20
records_per_class = 100

fzm_img = image.load_img(fzm_path, target_size=(224,224))
x = image.img_to_array(fzm_img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)


#prediction
preds = module.predict(x)








# We take only 2 classes from CIFAR10 and a very small sample to intentionally overfit the model.
# We will also use the same data for train/test and expect that Keras will give the same accuracy.
(x, y), _ = cifar10.load_data()
print(x.shape)
def filter_resize(category):
       # We do the preprocessing here instead in the Generator to get around a bug on Keras 2.1.5.
   return [preprocess_input(np.array(Image.fromarray(img).resize((224,224)))) for img in x[y.flatten()==category][:records_per_class]]

x = np.stack(filter_resize(3)+filter_resize(5))
records_per_class = x.shape[0] // 2
y = np.array([[1,0]]*records_per_class + [[0,1]]*records_per_class)


# We will use a pre-trained model and finetune the top layers.

np.random.seed(seed)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
last = Flatten()(base_model.output)
predictions = Dense(2, activation="softmax")(last)
finetuned_model = Model( base_model.input, predictions)

for layer in finetuned_model.layers[:140]:
    layer.trainable = False

for layer in finetuned_model.layers[140:]:
    layer.trainable = True

    
finetuned_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
finetuned_model.fit_generator(ImageDataGenerator().flow(x, y, seed=42), epochs=epochs, validation_data=ImageDataGenerator().flow(x, y, seed=42))
    
finetuned_model.save('resnet50_final.h5')

# In every test we will clear the session and reload the model to force Learning_Phase values to change.
print('DYNAMIC LEARNING_PHASE')
K.clear_session()
model = load_model('resnet50_final.h5')
# This accuracy should match exactly the one of the validation set on the last iteration.
print(model.evaluate_generator(ImageDataGenerator().flow(x, y, seed=42)))


print('STATIC LEARNING_PHASE = 0')
K.clear_session()
K.set_learning_phase(0)
model = load_model('resnet50_final.h5')
# Again the accuracy should match the above.
print(model.evaluate_generator(ImageDataGenerator().flow(x, y, seed=42)))


print('STATIC LEARNING_PHASE = 1')
K.clear_session()
K.set_learning_phase(1)
model = load_model('resnet50_final.h5')
# The accuracy will be close to the one of the training set on the last iteration.
print(model.evaluate_generator(ImageDataGenerator().flow(x, y, seed=42)))