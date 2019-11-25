import numpy as np
import csv
import cv2
import os

emotions = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
ferDirectory = 'fer2013'

def load_fer2013(filepath, counts=(28709, 3589, 3589)):
    reader = csv.reader(open(filepath))
    # skip header
    next(reader)

    training = []
    publicTest = []
    privateTest = []

    for row in reader:
        emotion = int(row[0])
        image = np.fromstring(row[1], dtype=np.dtype('float64'), sep=' ').reshape((48, 48))
        image = np.stack((image,)*3, axis=-1)
        if row[2] == 'Training' and len(training) < counts[0]:
            training.append((emotion, image))
        elif row[2] == 'PublicTest' and len(publicTest) < counts[1]:
            publicTest.append((emotion, image))
        elif row[2] == 'PrivateTest' and len(privateTest) < counts[2]:
            privateTest.append((emotion, image))
        if len(training) == counts[0] and len(publicTest) == counts[1] and len(privateTest) == counts[2]:
            break

    return training, publicTest, privateTest

def saveDatasetAsImages(dataset, outputDir):
    counts = np.zeros(7, dtype=int)
    for label, image in dataset:
        filename = '{}_{}.png'.format(emotions[label], counts[label])
        cv2.imwrite(os.path.join(outputDir, emotions[label], filename), image)
        counts[label] = counts[label] + 1

if __name__ == '__main__':
    training, publicTest, privateTest = load_fer2013(os.path.join(ferDirectory, 'fer2013.csv'))
    saveDatasetAsImages(training, os.path.join(ferDirectory, 'Training'))
    saveDatasetAsImages(publicTest, os.path.join(ferDirectory, 'PublicTest'))
    saveDatasetAsImages(privateTest, os.path.join(ferDirectory, 'PrivateTest'))
