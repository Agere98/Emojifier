import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--image', help='path to an image file to be classified')
    parser.add_argument('-c', '--classifier', help='path to a classifier', default='model.h5')
    args = parser.parse_args()

import cv2
import os
import numpy as np
from keras.models import Model, load_model
from face_extractor import getBoundingBox, extractFace
from trainer import emotions

emojis = {emotion: [] for emotion in emotions}

def process(image, classifier, extract_face=True):
    if extract_face:
        face = extractFace(image, getCentermost=True)
    else:
        face = image
    if face is not None:
        x = face.astype('float64')
        x[..., 0] -= 110.157036
        x[..., 1] -= 122.28291
        x[..., 2] -= 122.28291
        x = np.expand_dims(x, axis=0)
        predictions = classifier.predict(x)
        return list(zip(emotions, predictions[0]))
    return []

def predictImage(path, classifier):
    image = cv2.imread(path)
    predictions = process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), classifier)
    if predictions:
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        for label, value in predictions:
            print('{:>8.4f} {}'.format(value, label))
        drawBoundingBox(image, predictions)
        drawEmojis(image, predictions)
    cv2.imshow('Emojifier', image)
    cv2.waitKey()

def predictRealTime(classifier):
    cv2.namedWindow('Emojifier')
    camera = cv2.VideoCapture(0)
    while True:
        frame = camera.read()[1].copy()
        predictions = process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), classifier)
        if predictions:
            drawBoundingBox(frame, predictions)
            drawEmojis(frame, predictions)
            showProbabilities(predictions)
        cv2.imshow('Emojifier', frame)
        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

def drawBoundingBox(image, predictions=None):
    (bx, by, bw, bh) = getBoundingBox(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), getCentermost=True)
    cv2.rectangle(image, (bx, by), (bx+bw, by+bh), (0, 128, 255), 2)
    if predictions:
        label = sorted(predictions, key=lambda x: x[1], reverse=True)[0][0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, label, (bx, by-10), font, 0.5, (0, 128, 255), 1)

def showProbabilities(predictions):
    probabilities = np.zeros((260, 336, 3), dtype="uint8")
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (emotion, prob) in enumerate(predictions):
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        x, y, w, h = 8, 36*i+8, int(320*prob), 32
        cv2.rectangle(probabilities, (x, y), (x+w, y+h), (255, 0, 0), -1)
        cv2.putText(probabilities, text, (x+8, y+16), font, 0.5, (255, 255, 255), 1)
    cv2.imshow('Probabilities', probabilities)

def overlay(image, overlay, offset=(0, 0)):
    alpha = overlay[..., 3] / 255
    h, w = alpha.shape
    for c in range(3):
        color = overlay[..., c] * (alpha)
        beta  = image[offset[1]:offset[1]+h, offset[0]:offset[0]+w, c] * (1 - alpha)
        image[offset[1]:offset[1]+h, offset[0]:offset[0]+w, c] = color + beta

def drawEmojis(image, predictions=None):
    if predictions is not None:
        height, width, _ = image.shape
        size = min(width//4, height//5)
        size_ = int(0.8*size)
        top = height - size
        top_ = height - size_
        center = width // 2
        slots = [(center-size, top), (center, top), (center-size-size_, top_), (center+size, top_)]
        labels = sorted(predictions, key=lambda x: x[1], reverse=True)[:3]
        icons = [cv2.resize(emojis[labels[0][0]][0], (size, size)),
                 cv2.resize(emojis[labels[0][0]][1], (size, size)),
                 cv2.resize(emojis[labels[1][0]][1], (size_, size_)),
                 cv2.resize(emojis[labels[2][0]][1], (size_, size_))]
        cv2.rectangle(image, (0, top), (width, height), (0, 0, 0), -1)
        for icon, slot in zip(icons, slots):
            overlay(image, icon, slot)

def loadEmojis():
    icons = { 'neutral':  ['1F642.png', '1F610.png'],
              'happy':    ['1F603.png', '1F602.png'],
              'sad':      ['1F612.png', '2639.png'],
              'surprised':['1F62E.png', '1F928.png'],
              'angry':    ['1F620.png', '1F92C.png'],
              'disgust':  ['1F623.png', '1F92E.png'],
              'scared':   ['1F631.png', '1F632.png'] }
    for emotion in icons:
        for img in icons[emotion]:
            image = cv2.imread(os.path.join('Emoji', img), -1)
            emojis[emotion].append(image)

def main():
    global args
    modelPath = args.classifier
    model = load_model(modelPath)
    loadEmojis()
    if args.image != None:
        predictImage(args.image, model)
    else:
        predictRealTime(model)

if __name__ == '__main__':
    main()
