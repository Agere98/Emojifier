from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input
from keras.preprocessing.image import ImageDataGenerator
from keras_vggface.vggface import VGGFace
from data_loader import load_fer2013
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', help='number of epochs to train the model', type=int, default=3)
args = parser.parse_args()

def train(model):
    global args
    num_epochs = args.num_epochs
    # TODO
    return model

def getCleanModel():
    nb_class = 7
    vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
    return Model(vgg_model.input, out)

def main():
    model = getCleanModel()
    print('Inputs: {}'.format(model.inputs))
    print('Outputs: {}'.format(model.outputs))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model = train(model)
    model.save('model.h5')

if __name__ == '__main__':
    main()