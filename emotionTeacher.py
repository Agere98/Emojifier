from keras.utils import to_categorical
from keras.engine import Model
from keras.layers import Dense, Flatten
from keras_vggface.vggface import VGGFace
from skimage.transform import resize
import numpy as np
import pickle


def load_data(start, end):
    data = np.genfromtxt('fer2013/fer2013.csv', delimiter=',', dtype=None, encoding=None)
    labels = data[1:, 0].astype(np.int32)
    images = data[1:, 1]
    print("załadowano")
    return images[start:end], labels[start:end] #  (imgs_train, labels_train), (imgs_test, labels_test)


def add_copies(images1):
    images1 = np.array(list(zip(images1, images1.copy(), images1.copy())))
    images1 = np.einsum('abcd->acdb', images1)
    return images1


def divide_to_train_and_test(images, labels):
    count = len(images)
    board = int(count * 0.8)
    imgs_train = images[:board]
    labels_train = labels[:board]
    imgs_test = images[board:]
    labels_test = labels[board:]  # 28709
    return (imgs_train, labels_train), (imgs_test, labels_test)


def preprocess_data(images, labels):
    images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in images])
    images = np.array([np.reshape(image, (48, 48)) for image in images])
    print("zmieniono kształt")
    images = np.array([resize(image, (224, 224)) for image in images])
    images = images.astype('float32')
    print("zmieniono wielkość")
    #  (images_train, labels_train), (images_test, labels_test) = divide_to_train_and_test(images, labels)
    print("podzielono")
    images = add_copies(images)
    print("Dodano kopie")
    # one-hot encode target column
    labels = to_categorical(labels)  # change digit into a binary matrix representation
    print("Zmieniono oznaczenia na macierze binarne")
    return (images, labels)


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


def save_model(model):
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))


def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


if __name__ == '__main__':
    # Prepare testing dataset
    (images_test, answer_test) = load_data(30000, 31000)
    (images_test, answer_test) = preprocess_data(images_test, answer_test)
    model = load_model('finalized_model.sav')
    # model = get_vgg_model()# change this on load our model, when we will create ours
    # prepare model for training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    for i in range(30):  # divide into 30 datasets, because python have memory restrictions
        (images_train, answer_train) = load_data(i * 1000, i * 1000 + 1000)
        (images_train, answer_train) = preprocess_data(images_train, answer_train)
        # training model
        model.fit(images_train, answer_train, validation_data=([images_test, answer_test]), epochs=3)
    save_model(model)
    #save our model here
