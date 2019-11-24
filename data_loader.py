import numpy as np
import csv
import cv2

emotions = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']

def load_fer2013(filepath):
    reader = csv.reader(open(filepath))
    # skip header
    next(reader)

    training = []
    publicTest = []
    privateTest = []

    for row in reader:
        emotion = int(row[0])
        image = np.fromstring(row[1], dtype=np.dtype('float64'), sep=' ').reshape((48, 48))
        if row[2] == 'Training':
            training.append((emotion, image))
        elif row[2] == 'PublicTest':
            publicTest.append((emotion, image))
        elif row[2] == 'PrivateTest':
            privateTest.append((emotion, image))

    return training, publicTest, privateTest

def preprocessData(dataset):
    labels, images = zip(*dataset)
    resized = []
    for image in images:
        image = np.stack((image,)*3, axis=-1)
        image = cv2.resize(image, (224, 224))
        resized.append(image)
    return labels, resized

if __name__ == '__main__':
    training, _, _ = load_fer2013('fer2013/fer2013.csv')
    labels, images = preprocessData(training[0:10])
    for lb, im in zip(labels, images):
        cv2.imshow(emotions[lb], im/255)
        k = cv2.waitKey()
        cv2.destroyWindow(emotions[lb])
        if k == 27 or k == ord('q'):
            break