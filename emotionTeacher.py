from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.engine import Model
from keras.layers import Dense, Conv2D, Flatten, Input
from keras_vggface.vggface import VGGFace
from PIL import Image
from skimage.transform import resize
import numpy as np
def load_data():
    # download mnist data and split into train and test sets( !! mnist is for digits recognistion)
    # dataset = mnist.load_data() - just give it to return if test on this dataset
    data = np.genfromtxt('fer2013/fer2013.csv', delimiter=',', dtype=None, encoding=None)
    labels = data[1:, 0].astype(np.int32)
    image_buffer = data[1:, 1]
    images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in image_buffer])
    images = np.array([np.reshape(image, (48, 48)) for image in images])
    print("zmieniono kształt")
    images = np.array([resize(image, (224, 224)) for i, image in enumerate(images)]) # trwa bardzo długo, wiec sie przygotowac
    print("zmieniono wielkość")
    imgs_train = images[:28709]
    labels_train = labels[:28709]
    imgs_test = images[28709:]
    labels_test = labels[28709:]
    print("załadowano")
    return (imgs_train, labels_train), (imgs_test, labels_test)


def chunkIt(seq, n):  # divide list into n sized sublists
    out = []
    last = 0
    while last < len(seq):
        out.append(seq[last:(last + n)])
        last += n
    return out


def reshape_imgs(images1, images2):
    # reshape data to fit model
    images1 = images1.reshape(len(images1), 48, 48, 1)  # reshape 60000 images. 1 mean grayscale
    images2 = images2.reshape(len(images2), 48, 48, 1)#224 resize
    return images1, images2


def add_copies(images1, images2):
    images1 = zip(images1, images1.copy(), images1.copy())
    images2 = zip(images2, images2.copy(), images2.copy())
    return images1, images2


def preprocess_data():
    (images_train, labels_train), (images_test, labels_test) = load_data()
    plt.imshow(images_train[0])
    # images_train, images_test = reshape_imgs(images_train, images_test)
    images_train, images_test = add_copies(images_train, images_test)

    # one-hot encode target column
    labels_train = to_categorical(labels_train)  # change digit into a binary matrix representation
    labels_test = to_categorical(labels_test)
    return (images_train, labels_train), (images_test, labels_test)


def get_vgg_model():  # extract inner layers to train
    nb_class = 2
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
    custom_vgg_model = Model(vgg_model.input, out)
    return custom_vgg_model


if __name__ == '__main__':
    (images_train, answer_train), (images_test, answer_test) = preprocess_data()
    print(images_train[0])
    print(answer_train)
    model = get_vgg_model()

    # prepare model for training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # training model
    model.fit(images_train, answer_train, validation_data=(images_test, answer_test), epochs=1)
