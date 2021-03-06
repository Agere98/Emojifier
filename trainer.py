import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name under which the model will be saved')
    parser.add_argument('--num_epochs', help='number of epochs to train the model', type=int, default=1)
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--load', help='filepath to an already trained model to initialize the neural network before training', dest='filepath')
    parser.add_argument('-d', '--dataset', help='root directory of a training dataset', default='dataset')
    parser.add_argument('-b', '--save_best', help='if specified, the current best model is saved to a separate file', action='store_true')
    parser.add_argument('-v', '--verbose', help='set verbosity mode', action='count', default=0)
    parser.add_argument('-s', '--summary', help='show model summary and exit', action='store_true')
    args = parser.parse_args()

from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras_vggface.vggface import VGGFace
import numpy as np
import cv2
import os

emotions = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']

def train(model, datasetDir, num_epochs=1, batch_size=32, verbosity=0, checkpoint_dir=None, log_dir=None):

    if verbosity > 0:
        print('Preparing training data...')

    trainDatagen = ImageDataGenerator(
        featurewise_center=True,
        horizontal_flip=True,
        rotation_range=35,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.4, 1.5),
        zoom_range=0.05,
        fill_mode='nearest')
    
    sample = getSample(100, datasetDir)
    trainDatagen.fit(sample)

    validationDatagen = ImageDataGenerator(
        featurewise_center=True,
        horizontal_flip=True)
    
    validationDatagen.fit(sample)

    if verbosity > 0:
        print('Mean values for preprocessing: {}'.format(validationDatagen.mean))

    train_generator = trainDatagen.flow_from_directory( 
        os.path.join(datasetDir, 'training'), 
        target_size=(224, 224), 
        batch_size=batch_size, 
        class_mode='categorical',
        classes=emotions,
        interpolation='bilinear')

    validation_generator = validationDatagen.flow_from_directory(
        os.path.join(datasetDir, 'validation'), 
        target_size=(224, 224), 
        batch_size=batch_size, 
        class_mode='categorical',
        classes=emotions,
        interpolation='bilinear')

    callbacks = []
    if checkpoint_dir:
        filepath = os.path.join(checkpoint_dir, 'best_{epoch:02d}.h5')
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max')
        callbacks.append(checkpoint)
    if log_dir:
        tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, update_freq=1000)
        callbacks.append(tensorboard)

    if verbosity > 0:
        print('Starting training...')
    
    model.fit_generator(
        train_generator,
        epochs=num_epochs,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        verbose=verbosity,
        callbacks=callbacks)
        
    return model

def getSample(classCount, datasetDir, subset='training'):
    datasetDir = os.path.join(datasetDir, subset)
    sample = []
    for emotion in emotions:
        path = os.path.join(datasetDir, emotion)
        if os.path.exists(path):
            count = 0
            for filename in os.listdir(path):
                image = cv2.imread(os.path.join(path, filename))
                if image is None or image.size == 0:
                    continue
                sample.append(image)
                count += 1
                if count == classCount:
                    break
    return sample

def getCleanModel():
    nb_class = 7
    vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_model.layers[:-11]:
        layer.trainable = False
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='dense')(x)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
    model = Model(vgg_model.input, out)
    return model

def main():
    global args
    clip = lambda x, l, u: l if x < l else u if x > u else x
    verbosity = clip(args.verbose, 0, 2)

    if verbosity > 0:
        print('Initializing model...')
    if args.filepath == None:
        model = getCleanModel()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model = load_model(args.filepath)
    if verbosity > 0:
        print('Inputs: {}'.format(model.inputs))
        print('Outputs: {}'.format(model.outputs))
    if args.summary:
        model.summary()
        exit()
    modelDir = os.path.join('models', args.name)
    os.makedirs(modelDir, exist_ok=True)
    logDir = os.path.join('logs', args.name)
    os.makedirs(logDir, exist_ok=True)
    checkpointDir = None
    if args.save_best:
        checkpointDir = modelDir
    if verbosity > 0:
        print('Model initialized.')

    model = train(model, args.dataset, args.num_epochs, args.batch_size, verbosity=verbosity, 
        checkpoint_dir=checkpointDir, log_dir=logDir)

    if verbosity > 0:
        print('Training completed. Saving model...')
    model.save(os.path.join(modelDir, '{}.h5'.format(args.name)))
    if verbosity > 0:
        print('Model saved.')

if __name__ == '__main__':
    main()
