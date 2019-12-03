from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from face_extractor import extractFace
from trainer import getSample
from emojifier import process
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
        featurewise_center=True)
    sample = getSample(100, os.path.join(datasetDir, 'training'))
    evaluateDatagen.fit(sample)
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


def predictImage(path, model):
    image = cv2.imread(path)
    preds = process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), model)
    if preds:
        preds = sorted(preds, key=lambda x: x[1], reverse=True)
        for label, value in preds:
            print('{:>8.4f} {}'.format(value, label))
        print('')
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


