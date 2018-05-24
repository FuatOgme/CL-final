import foolbox
from keras.datasets import cifar10
import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
from keras.models import load_model
import os


def plotImgs(image, adversarial, predict_label, test_label):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('Original ' + str(predict_label) + ' <> ' + str(test_label))
    plt.imshow(image / 255)  # division by 255 to convert [0, 255] to [0, 1]
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial')
    plt.imshow(adversarial[:, :, ::-1] / 255)  # ::-1 to convert BGR to RGB
    plt.axis('off')

    plt.subplot(1, 3, 3)
    difference = adversarial[:, :, ::-1] - image
    sum = np.ndarray.sum(difference)
    plt.title('Difference(' + str(sum) + ')')
    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
    plt.axis('off')

    plt.show()

#Params
normalized = False
numClasses = 10

# kaydedilmiş model yükleniyor.
model_name = 'keras_cifar10_trained_epc100_model.h5'
load_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = os.path.join(load_dir, model_name)
model = load_model(model_path)

# instantiate model
# keras.backend.set_learning_phase(0)
# kmodel = ResNet50(weights='imagenet')
# preprocessing = (np.array([104, 116, 123]), 1)
# fmodel = foolbox.models.KerasModel(model, bounds=(0, 255), preprocessing=preprocessing)

#cifar10
(x_org_train, y_org_train), (x_org_test, y_org_test) = cifar10.load_data()

# if normalized:
#     x_train = x_org_train.astype('float32')
#     x_test = x_org_test.astype('float32')
#     x_train /= 255
#     x_test /= 255

# y_test = keras.utils.to_categorical(y_org_test, numClasses)
# scores = model.evaluate(x_test, y_test, verbose=1)
# fmodel = foolbox.models.KerasModel(model, bounds=(0, 1) if normalized else (0,255))

fmodel = foolbox.models.KerasModel(model, bounds=(0,255))




x_port_test = x_org_test[:40]
y_port_test = y_org_test[:40]


# scores = model.evaluate(smallX, keras.utils.to_categorical(smallY), verbose=1)
# res = model.predict(x_org_test[:40],verbose=1)
res_classes = model.predict_classes(x_port_test[:40],verbose=1)

# apply attack on source image
# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
attack = foolbox.attacks.FGSM(fmodel)

for i in range(0,10):
    image = x_org_test[i]
    label = y_org_test[i].max()
    adversarial = attack(image[:, :, ::-1], label)
    plotImgs(image,adversarial,res_classes[i], label)

T = 5


# kaydedilmiş model yükleniyor.
# model_name = 'keras_cifar10_trained_model.h5'
# load_dir = os.path.join(os.getcwd(), 'saved_models')
# model_path = os.path.join(load_dir, model_name)
# model = load_model(model_path)


#
# attack = foolbox.attacks.FGSM(model)
# adversarial = attack(image[:, :, ::-1], label)


