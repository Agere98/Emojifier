from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras_vggface.vggface import VGGFace
import numpy as np
import argparse
import os
from data_loader import load_fer2013, emotions, ferDirectory

parser = argparse.ArgumentParser()
parser.add_argument('name', help='name under which the model will be saved')
parser.add_argument('--num_epochs', help='number of epochs to train the model', type=int, default=1)
parser.add_argument('--batch_size', help='batch size', type=int, default=32)
parser.add_argument('--load', help='filepath to an already trained model to initialize the neural network before training', dest='filepath')
parser.add_argument('--save_best', help='if specified, the current best model is saved to a separate file', action='store_true')
parser.add_argument('-v', '--verbose', help='set verbosity mode', action='count', default=0)
args = parser.parse_args()

def train(model, num_epochs=1, batch_size=32, verbosity=0, checkpoint_dir=None):

    if verbosity > 0:
        print('Preparing training data...')

    trainDatagen = ImageDataGenerator(
        featurewise_center=True,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.1,
        horizontal_flip=True)
    
    sample, _, _ = load_fer2013(os.path.join(ferDirectory, 'fer2013.csv'), counts=(1000, 0, 0))
    _, sample = zip(*sample)
    sample = np.array(sample)
    trainDatagen.fit(sample)

    validationDatagen = ImageDataGenerator(
        horizontal_flip=True)

    train_generator = trainDatagen.flow_from_directory( 
        os.path.join(ferDirectory, 'Training'), 
        target_size=(224, 224), 
        batch_size=batch_size, 
        class_mode='categorical',
        classes=emotions,
        interpolation='bilinear')

    validation_generator = validationDatagen.flow_from_directory(
        os.path.join(ferDirectory, 'PublicTest'), 
        target_size=(224, 224), 
        batch_size=batch_size, 
        class_mode='categorical',
        classes=emotions,
        interpolation='bilinear')

    callbacks = None
    if checkpoint_dir:
        filepath = os.path.join(checkpoint_dir, 'best_{epoch:02d}-{val_accuracy:.2f}.h5')
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')
        callbacks = [checkpoint]

    if verbosity > 0:
        print('Starting training...')
    
    model.fit_generator(
        train_generator,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        verbose=verbosity,
        callbacks=callbacks)
        
    return model

def getCleanModel():
    nb_class = 7
    vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
    return Model(vgg_model.input, out)

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
        model = load_model(filepath)
    if verbosity > 0:
        print('Inputs: {}'.format(model.inputs))
        print('Outputs: {}'.format(model.outputs))
    modelDir = os.path.join('models', args.name)
    os.makedirs(modelDir, exist_ok=True)
    checkpointDir = None
    if args.save_best:
        checkpointDir = modelDir
    if verbosity > 0:
        print('Model initialized.')

    model = train(model, args.num_epochs, args.batch_size, verbosity=verbosity, checkpoint_dir=checkpointDir)

    if verbosity > 0:
        print('Training completed. Saving model...')
    model.save(os.path.join(modelDir, '{}.h5'.format(args.name)))
    if verbosity > 0:
        print('Model saved.')

if __name__ == '__main__':
    main()
