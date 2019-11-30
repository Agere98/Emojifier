import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--image', help='path to an image file to be classified')
    parser.add_argument('-c', '--classifier', help='path to a classifier', default='model.h5')
    args = parser.parse_args()

import cv2
import numpy as np
from keras.models import Model, load_model
from face_extractor import getBoundingBox, extractFace
from trainer import emotions

def process(image, classifier):
    face = extractFace(image)
    if face is not None:
        x = face.astype('float64')
        x[..., 0] -= 110.157036
        x[..., 1] -= 122.28291
        x[..., 2] -= 122.28291
        x = np.expand_dims(x, axis=0)
        preds = classifier.predict(x)
        return list(zip(emotions, preds[0]))
    return []

def predictImage(path, classifier):
    image = cv2.imread(path)
    preds = process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), classifier)
    if preds:
        preds = sorted(preds, key=lambda x: x[1], reverse=True)
        for label, value in preds:
            print('{:>8.4f} {}'.format(value, label))
        drawBoundingBox(image, preds)
    cv2.imshow("Emojifier", image)
    cv2.waitKey()

def predictRealTime(classifier):
    cv2.namedWindow('Emojifier')
    camera = cv2.VideoCapture(0)
    while True:
        frame = camera.read()[1].copy()
        preds = process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), classifier)
        if preds:
            drawBoundingBox(frame, preds)
            showProbabilities(preds)
        cv2.imshow('Emojifier', frame)
        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

def drawBoundingBox(image, predictions=None):
    (bx, by, bw, bh) = getBoundingBox(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
    cv2.imshow("Probabilities", probabilities)

def main():
    global args
    modelPath = args.classifier
    model = load_model(modelPath)
    if args.image != None:
        predictImage(args.image, model)
    else:
        predictRealTime(model)

if __name__ == '__main__':
    main()
