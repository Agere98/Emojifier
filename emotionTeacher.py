from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.engine import Model
from keras.layers import Dense, Flatten
from keras_vggface.vggface import VGGFace
from skimage.transform import resize
import numpy as np
import cv2

def load_data():
    # download mnist data and split into train and test sets( !! mnist is for digits recognistion)
    # dataset = mnist.load_data() - just give it to return if test on this dataset
    data = np.genfromtxt('fer2013/fer2013.csv', delimiter=',', dtype=None, encoding=None)
    labels = data[1:, 0].astype(np.int32)
    images = data[1:, 1]
    print("załadowano")
    return images[:320], labels[:320] #  (imgs_train, labels_train), (imgs_test, labels_test)


def add_copies(images1, images2):
    images1 = np.array(list(zip(images1, images1.copy(), images1.copy())))
    images2 = np.array(list(zip(images2, images2.copy(), images2.copy())))
    images1 = np.einsum('abcd->acdb', images1)
    images2 = np.einsum('abcd->acdb', images2)
    return images1, images2


def divide_to_train_and_test(images, labels):
    count = len(images)
    board = int(count * 0.8)
    imgs_train = images[:board]
    labels_train = labels[:board]
    imgs_test = images[board:]
    labels_test = labels[board:]  # 28709
    return (imgs_train, labels_train), (imgs_test, labels_test)


def preprocess_data():
    images, labels = load_data()
    images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in images])
    images = np.array([np.reshape(image, (48, 48)) for image in images])
    print("zmieniono kształt")
    images = np.array([resize(image, (224, 224)) for image in images])
    images = images.astype('float32')
    print("zmieniono wielkość")
    (images_train, labels_train), (images_test, labels_test) = divide_to_train_and_test(images, labels)
    print("podzielono")
    images_train, images_test = add_copies(images_train, images_test)
    print("Dodano kopie")
    # one-hot encode target column
    labels_train = to_categorical(labels_train)  # change digit into a binary matrix representation
    labels_test = to_categorical(labels_test)
    print("Zmieniono oznaczenia na macierze binarne")
    return (images_train, labels_train), (images_test, labels_test)


def get_vgg_model():  # extract inner layers to train
    nb_class = 7
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    # for layer in vgg_model.layers:
    #    print(layer.name)
    #  in documentation was avg_pool, but there is no layer named avg_pool. Took layer named pool3 from the middle
    last_layer = vgg_model.get_layer('pool3').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
    custom_vgg_model = Model(vgg_model.input, out)
    return custom_vgg_model


if __name__ == '__main__':
    (images_train, answer_train), (images_test, answer_test) = preprocess_data()
    model = get_vgg_model()
    # prepare model for training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # training model
    model.fit(images_train, answer_train, validation_data=([images_test, answer_test]), epochs=3)
