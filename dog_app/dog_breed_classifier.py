import os
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pylab as plt

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.applications.resnet50 import ResNet50, preprocess_input

from face_detector import FaceDetector

DEFAULT_BATCH_SIZE = 20
DEFAULT_EPOCHS = 20
DEFAULT_SIZE = (244, 244)


def ResNet50_predict_labels(img_path):
    ResNet50_model = ResNet50(weights='imagenet')
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def extract_Resnet50(tensor):
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=DEFAULT_SIZE)

    x = image.img_to_array(img)

    return np.expand_dims(x, axis=0)


def predict_dog_breed_ResNet50(model, img_path):
    """
    This solution is basically identical to the VGG16 version
    Not sure what else was necessary here?
    """

    dog_names = load_dog_names('data/dogs_list.txt')

    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))

    # Make predictions
    predicted_vector = model.predict(bottleneck_feature)

    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def load_dog_names(fname):
    fh = open(fname)
    dog_names = [line[4:].rstrip().replace('_', ' ') for line in fh]
    fh.close()
    return dog_names


def show_image(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def my_predict_breed(model, img_path):
    img = plt.imread(img_path)

    facedector = FaceDetector()

    faces = facedector.detect_faces(image_path=img_path, show=False, return_img=False)

    if len(faces) > 0:
        print('Hey there, human!')
        prediction_text = 'You look like a {}'

    elif dog_detector(img_path):
        print('Hey there, doggo!')
        prediction_text = 'Your predicted breed is a {}'

    else:
        print('ERROR: This app only works when given images of Humans or doggos...')
        print('Given image:')
        show_image(img)
        return

    predicted_breed = predict_dog_breed_ResNet50(model, img_path=img_path)

    show_image(img)
    print(prediction_text.format(predicted_breed))

if __name__ == '__main__':
    features = np.load('data/bottleneck_features/DogResnet50Data.npz')
    train_features = features['train']
    valid_features = features['valid']
    test_features = features['test']

    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
    model.add(Dropout(0.2))
    model.add(Dense(133, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights('data/saved_models/weights.best.ResNet50.hdf5')

    my_predict_breed(model, 'data/test_images/me.jpg')
