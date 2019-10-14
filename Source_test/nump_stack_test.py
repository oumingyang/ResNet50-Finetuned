import numpy as np
from keras.datasets import cifar10
from keras.applications.resnet50 import preprocess_input
from PIL import Image

arrays = [np.random.rand(3,4) for _ in range(10)]
arrays_axis0 = np.stack(arrays, axis=0)
arrays_shape_axis0 = arrays_axis0.shape
print(arrays_axis0)
print(arrays_shape_axis0)

arrays_axis1 = np.stack(arrays, axis=1)
arrays_shape_axis1 = arrays_axis1.shape
print(arrays_axis1)
print(arrays_shape_axis1)

arrays_axis2 = np.stack(arrays, axis=2)
arrays_shape_axis2 = arrays_axis2.shape
print(arrays_axis2)
print(arrays_shape_axis2)


records_per_class = 300

(x, y), _ = cifar10.load_data()
print(x.shape)
print(y.shape)
print(y.flatten())

def filter_resize(category):
       # We do the preprocessing here instead in the Generator to get around a bug on Keras 2.1.5.
   return [preprocess_input(np.array(Image.fromarray(img).resize((224,224)))) for img in x[y.flatten()==category][:records_per_class]]

x = np.stack(filter_resize(3)+filter_resize(5))
print(x.shape)
records_per_class = x.shape[0] // 2
y = np.array([[1,0]]*records_per_class + [[0,1]]*records_per_class)
print(records_per_class)
print(y.shape)
print(y)
