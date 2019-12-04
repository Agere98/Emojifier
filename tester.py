import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='filepath to a model to be loaded for evaluation', default='model_acc.h5')
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('-d', '--dataset', help='root directory of a training dataset', default='dataset')
    args = parser.parse_args()

import numpy as np
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from trainer import getSample, emotions
from emojifier import process
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def test_model(model, datasetDir=args.dataset, batch_size=args.batch_size):

    evaluateDatagen = ImageDataGenerator(
        featurewise_center=True)
    sample = getSample(100, datasetDir)
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
        print('Validation set directory not found.')


def predictImage(path, model):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preds = process(image, model, extract_face=False)
    if preds:
        preds = sorted(preds, key=lambda x: x[1], reverse=True)
        best_label, _ = preds[0]
        return best_label
    return None


def test_confusion_matrix(img_dir, model):
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

    matrix = confusion_matrix(label_true, label_pred, labels=emotions, normalize='true')
    print(matrix)
    df_cm = pd.DataFrame(matrix, index=[i for i in emotions],
                         columns=[i for i in emotions])
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True)
    plt.show()
    plt.imsave('matrix.jpg')


def main():
    global args
    model = load_model(args.model)
    test_model(model)
    #test_confusion_matrix(os.path.join(args.dataset, 'validation'), model)

if __name__ == "__main__":
    main()
