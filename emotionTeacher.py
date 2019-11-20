from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.engine import Model
from keras.layers import Dense, Conv2D, Flatten, Input
from keras_vggface.vggface import VGGFace
from PIL import Image
import numpy as np
def load_data():
    # download mnist data and split into train and test sets( !! mnist is for digits recognistion)
    # jaffedbase has 213 images of females from asia
    # dataset = mnist.load_data() - just give it to return if test on this dataset
    imgs_train = []
    imgs_test = []
    answer_train = []
    answer = []
    emotions_labels = ['AN', 'FE', 'NE', 'SA', 'DI', 'SU', 'HA']
    for i, label in enumerate(emotions_labels):
        imgs = Image.open("jaffedbase/*.tiff")
        imarray = np.array(imgs)
        imgs_count = range(imarray)
        emotion = [i]
        answer = emotion * imgs_count
        imgs_train.append(imarray[:imgs_count * 5 / 6])
        answer_train.append(answer[:imgs_count * 5 / 6])
        imgs_test.append(imarray[imgs_count * 5 / 6:])
        answer_test.append(answer[imgs_count * 5 / 6:])
    return (imgs_train, answer_train), (imgs_test, answer_test)

def reshape_imgs(images1, images2):
    # reshape data to fit model
    images1 = images1.reshape(range(images1), 224, 224, 1)  # reshape 60000 images. 1 mean grayscale
    images2 = images2.reshape(range(images2), 224, 224, 1)
    return images1, images2


def preprocess_data():
    (images_train, answer_train), (images_test, answer_test) = load_data()
    plt.imshow(images_train[0])
    images_train, images_test = reshape_imgs(images_train, images_test)

    # one-hot encode target column
    answer_train = to_categorical(answer_train)  # change digit into a binary matrix representation
    answer_test = to_categorical(answer_test)
    return (images_train, answer_train), (images_test, answer_test)


def get_vgg_model():  # extract inner layers to train
    nb_class = 2
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
    custom_vgg_model = Model(vgg_model.input, out)
    return custom_vgg_model


if __name__ == '__main__':
    (images_train, answer_train), (images_test, answer_test) = load_data()
    print(images_train[0])
    # model = get_vgg_model()

    # prepare model for training
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # training model
    #odel.fit(images_train, answer_train, validation_data=(images_test, answer_test), epochs=3)
