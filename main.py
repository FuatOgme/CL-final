import keras
from keras.datasets import cifar10
import os
from keras.models import load_model


num_classes = 10

#kaydedilmiş model yükleniyor.
model_name = 'keras_cifar10_trained_model.h5'
load_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = os.path.join(load_dir, model_name)
model = load_model(model_path)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


x = 5;