from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras_vggface.vggface import VGGFace
import numpy as np
import cv2
import os
import argparse

emotions = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', type=int, default=10)
    parser.add_argument('-l', '--load', help='filepath to an already trained model to initialize the neural network before training', default='model.h5')
    parser.add_argument('-d', '--dataset', help='root directory of a training dataset', default='dataset')
    parser.add_argument('-v', '--verbose', help='set verbosity mode', action='count', default=1)
    args = parser.parse_args()


def test_model(model, datasetDir=args.dataset, batch_size=args.batch_size):

    evaluateDatagen = ImageDataGenerator(
        featurewise_center=True,
        horizontal_flip=True,
        rotation_range=35,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.4, 1.5),
        zoom_range=0.05,
        fill_mode='nearest')

    path = os.path.join(datasetDir, 'test')
    if os.path.exists(path):
        test_generator = evaluateDatagen.flow_from_directory(
                path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='categorical',
                classes=emotions,
                interpolation='bilinear')

        result = model.evaluate_generator(test_generator)
        print(model.metrics_names[0] + '=' + str(round(result[0], 2)))
        print(model.metrics_names[1] + '=' + str(round(result[1] * 100, 2)) + '%')
    else:
        print('problem')


if __name__ == "__main__":
    model = load_model(args.load)
    test_model(model)


