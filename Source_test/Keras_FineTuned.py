import math, json, os, sys
import numpy as np
import _pickle as cPickle

import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, Flatten
from keras.models import Model, load_model

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras import backend as K


#Get current file path
current_path = os.path.abspath(__file__)

#Get current file father_dir
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
source_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Source_image")
mosaic_image_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Mosaic_image")
dataset_train_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Data_train")
dataset_valiate_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),"Data_valiate")

#Path of data and datasets

image_train_path = source_image_path
dataset_train_save_path = dataset_train_path

epochs = 20
records_per_class = 30


# We take only 2 classes from CIFAR10 and a very small sample to intentionally overfit the model.
# We will also use the same data for train/test and expect that Keras will give the same accuracy.

def unpickle(file):
    with open(file, 'rb') as fo:
        dataset_dict = cPickle.load(fo)
    return dataset_dict

dataset_unpickled = unpickle(dataset_train_save_path + os.path.sep + 'data_batch_0')
data = dataset_unpickled["data"]
label = dataset_unpickled["labels"]
y_array = np.array(label)
y_array = y_array.reshape((len(y_array),1))
x_reshape = data.reshape((len(data), 32, 32, 3))


def filter_resize(category):
    # We do the preprocessing here instead in the Generator to get around a bug on Keras 2.1.5.
    return [preprocess_input(np.array(Image.fromarray(img).resize((224,224)))) for img in x_reshape[y_array.flatten() == category][:records_per_class]]
# x = np.stack(filter_resize(1)+filter_resize(5))
# records_per_class = x.shape[0] // 2
# y = np.array([[1,0]]*records_per_class + [[0,1]]*records_per_class)
x = np.stack(filter_resize(1))
y = np.array([[1,0]]*records_per_class)

# We will use a pre-trained model and finetune the top layers.

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
last = Flatten()(base_model.output)
predictions = Dense(2, activation="softmax")(last)
finetuned_model = Model( base_model.input, predictions)

for layer in finetuned_model.layers[:140]:
    layer.trainable = False

for layer in finetuned_model.layers[140:]:
    layer.trainable = True

    
finetuned_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# batch_size = 32
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
