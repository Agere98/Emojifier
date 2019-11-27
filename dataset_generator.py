import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_dir', help='path to a directory where the dataset will be saved')
parser.add_argument('-l', '--query_limit', help='upper limit of images to retrieve from a single query', type=int)
parser.add_argument('-d', '--download', help='if specified, execute queries and download images', action='store_true')
parser.add_argument('-e', '--extract', help='if specified, extract faces from downloaded images', action='store_true')
args = parser.parse_args()
if args.output_dir is not None:
    args.download = True
if args.query_limit is not None:
    args.download = True
if not args.download and not args.extract:
    parser.print_usage()
    exit()

import cv2
import os
from mtcnn import MTCNN
from tqdm import tqdm
from google_images_download import google_images_download
from face_extractor import extractFace

def downloadFaces(queriesDict, outputDir, chromeDriverDir, queryLimit = 10):
    response = google_images_download.googleimagesdownload()
    args = {'keywords': '',
            'output_directory': outputDir,
            'image_directory': '',
            'silent_mode': True,
            'limit': queryLimit,
            'chromedriver': chromeDriverDir}
    for label, queries in queriesDict.items():
        for query in queries:
            args.update({'keywords': query, 'image_directory': label})
            response.download(args)

def filterFaces(sourceDir, destinationDir):
    for label in os.listdir(sourceDir):
        print('Extracting faces for: {}'.format(label))
        src = os.path.join(sourceDir, label)
        dst = os.path.join(destinationDir, label)
        os.makedirs(dst, exist_ok=True)
        for filename in tqdm(os.listdir(src)):
            image = cv2.imread(os.path.join(src, filename))
            if image is None or image.size == 0:
                continue
            face = extractFace(image, noneIfMultiple=True)
            if face is not None:
                filename = '{}.png'.format(os.path.splitext(filename)[0])
                cv2.imwrite(os.path.join(dst, filename), face)

def main():
    global args
    chromeDriverDir = os.path.join('..', 'chromedriver.exe')
    outputDir = args.output_dir if args.output_dir is not None else 'dataset'
    queryLimit = args.query_limit if args.query_limit is not None else 1000

    queries = { 'neutral':  ['neutral human face', 'neutral face expression'],
                'happy':    ['happy human face', 'person smiling'],
                'sad':      ['sad human face', 'person sad'],
                'surprised':['surprised human face', 'person amazed'],
                'angry':    ['angry human face', 'person furious'],
                'disgust':  ['disgusted human face', 'person disgust'],
                'scared':   ['scared human face', 'person fearful'] }
    
    imagesDir = os.path.join(outputDir, 'raw')
    if args.download:
        os.makedirs(imagesDir, exist_ok=True)
        downloadFaces(queries, imagesDir, chromeDriverDir, queryLimit=queryLimit)
    if args.extract:
        facesDir = os.path.join(outputDir, 'face')
        os.makedirs(facesDir, exist_ok=True)
        filterFaces(imagesDir, facesDir)

if __name__ == "__main__":
    main()
