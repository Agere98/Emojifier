from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras_vggface.vggface import VGGFace
from face_extractor import extractFace
import numpy as np
import cv2
import os
import argparse
from sklearn.metrics import confusion_matrix


emotions = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='batch size', type=int, default=10)
    parser.add_argument('-l', '--load', help='filepath to an already trained model to initialize the neural network before training', default='model_loss.h5')
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

    path = os.path.join(datasetDir, 'validation')
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


def process(image, model):
    face = extractFace(image)
    if face is not None:
        x = face.astype('float64')
        x[..., 0] -= 110.157036
        x[..., 1] -= 122.28291
        x[..., 2] -= 122.28291
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        return list(zip(emotions, preds[0]))
    return []


def predictImage(path, model):
    image = cv2.imread(path)
    preds = process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), model)
    if preds:
        preds = sorted(preds, key=lambda x: x[1], reverse=True)
        best_label, _ = preds[0]
        return best_label
    return None


def test_matrix_of_mistakes(img_dir, model):
    label_true = []
    label_pred = []
    for emotion in emotions:
        path = os.path.join(img_dir, emotion)
        listing = os.listdir(path)
        for name in listing:
            label = predictImage(os.path.join(path, name), model)
            if label:
                label_true.append(emotion)
                label_pred.append(label)

    matrix = confusion_matrix(label_true, label_pred, labels=['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised',
                                                              'neutral'])
    print(matrix)


if __name__ == "__main__":
    model = load_model(os.path.join('models', args.load))
    # test_model(model)
    test_matrix_of_mistakes(os.path.join(args.dataset, 'validation'), model)


