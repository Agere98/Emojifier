import cv2
import numpy as np
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--image', help='relative path to an image file to be classified')
args = parser.parse_args()

detector = MTCNN()
vggface = VGGFace(model='resnet50')

def getImage(path):
    image = cv2.imread(path)
    return image

def getBoundingBox(image):
    global detector
    faces = detector.detect_faces(image)
    if len(faces) > 0:
        return faces[0]['box']
    else:
        return None

def showBoundingBox(image, bb):
    if bb:
        cv2.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 0, 255), 2)
    cv2.imshow("Bounding box", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey()

def extractFace(image, bb, size=(224, 224)):
    image = image[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
    image = cv2.resize(image, size)
    return image

def process(image):
    global vggface
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bb = getBoundingBox(image)
    if bb:
        face = extractFace(image, bb)
        x = face.astype('float64')
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=2)
        preds = vggface.predict(x)
        print('Predicted:', utils.decode_predictions(preds))
    showBoundingBox(image, bb)

def main():
    global args
    if args.image != None:
        process(getImage(args.image))

if __name__ == '__main__':
    main()