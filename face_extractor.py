import cv2
from mtcnn import MTCNN

detector = MTCNN()

def getBoundingBox(image, noneIfMultiple=False):
    global detector
    faces = detector.detect_faces(image)
    if len(faces) == 1 or (len(faces) > 1 and not noneIfMultiple):
        return faces[0]['box']
    else:
        return None

def extractFace(image, size=(224, 224), noneIfMultiple=False):
    bb = getBoundingBox(image, noneIfMultiple)
    if bb is not None:
        image = image[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
        if image.size == 0:
            return None
        image = cv2.resize(image, size)
        return image
    return None
