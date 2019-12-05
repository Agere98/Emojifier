import cv2
from mtcnn import MTCNN

detector = MTCNN()

def getBoundingBox(image, noneIfMultiple=False, getCentermost=False):
    global detector
    faces = detector.detect_faces(image)
    if len(faces) == 1 or (len(faces) > 1 and not noneIfMultiple):
        if len(faces) == 1 or not getCentermost:
            return faces[0]['box']
        else:
            cy, cx, _ = image.shape
            cx, cy = cx // 2, cy // 2
            min = (cx + cy)**2
            f = 0
            print(len(faces))
            for i, face in enumerate(faces):
                x, y, w, h = face['box']
                bx, by = x + w // 2, y + h // 2
                d = (bx - cx)**2 + (by - cy)**2
                if d < min:
                    min = d
                    f = i
            return faces[f]['box']
    else:
        return None

def extractFace(image, size=(224, 224), noneIfMultiple=False, getCentermost=False):
    bb = getBoundingBox(image, noneIfMultiple, getCentermost)
    if bb is not None:
        image = image[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
        if image.size == 0:
            return None
        image = cv2.resize(image, size)
        return image
    return None
