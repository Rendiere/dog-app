"""
CNN-based dog breed classifier

Author: Renier Botha
Date: 23 Feb 2018
"""
import os
import io
import requests
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.applications.resnet50 import ResNet50, preprocess_input


class CNNModel():

    def __init__(self):
        # TODO: Export to config
        self.bottleneck_filepath = 'data/bottleneck_features/DogResnet50Data.npz'
        self.features_url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz'
        self.dogbreed_model_filepath = 'data/saved_models/weights.best.ResNet50.hdf5'
        self.dog_names_filepath = 'data/dogs_list.txt'

        self.image_size = (224, 224)

        # Load the necessary models and data
        self.Resnet50_model = self.load_resnet()
        self.bottleneck_features = self.load_bottleneck()
        self.dogbreed_model = self.load_dogbreed()
        self.dog_names = self.load_dog_names()

        self.graph = tf.get_default_graph()

    def load_resnet(self):
        print('Loading ResNet model...', end='')
        Resnet50_model = ResNet50(weights='imagenet', include_top=False)
        print('done')
        return Resnet50_model

    def load_bottleneck(self):
        print('Loading the bottleneck features')

        if os.path.isfile(self.bottleneck_filepath):
            features = np.load(self.bottleneck_filepath)
        else:
            from tqdm import tqdm
            print('Could not locate bottleneck features...')
            print('Downloading bottleneck features')

            # Download the file with a nice progress bar.
            req = requests.get(self.features_url, stream=True)

            file_size = int(req.headers.get('content-length', 0))

            with open(self.bottleneck_filepath, 'wb') as f:
                for data in tqdm(req.iter_content(32 * 1024), total=file_size, unit='kB', unit_scale=True, ncols=100):
                    f.write(data)

            # Now that it's downloaded, load into memory
            features = np.load(self.bottleneck_filepath)

        return features

    def load_dogbreed(self):
        # Load the model trained on dog breeds
        # TODO: Save the complete model and not only the weights
        # TODO: Add batch normalization
        print('Loading the dog breeds classifier...', end='')
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=self.bottleneck_features['train'].shape[1:]))
        model.add(Dropout(0.2))
        model.add(Dense(133, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(self.dogbreed_model_filepath)

        return model

    def load_dog_names(self):
        # Load the dog names
        fh = open(self.dog_names_filepath)
        dog_names = [line[4:].rstrip() for line in fh]
        fh.close()
        return dog_names

    def extract_Resnet50(self, image):
        """
        Get the bottleneck features from the "topless"
        ResNet50 model by running the image through one
        forward pass and returning the last layer

        :param image:
        :return:
        """
        return self.Resnet50_model.predict(image)

    def prepare_image(self, image):
        """
        Prepare the uploaded image for prediction

        :param image:
        :return:
        """

        image = image.resize(self.image_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        return preprocess_input(image)

    def predict(self, image):
        """
        TODO: Output probabilities
        TODO: Face recognition

        :param image:
        :return:
        """

        image = self.prepare_image(image)

        # extract bottleneck features
        bottleneck_feature = self.extract_Resnet50(image)

        # Make predictions
        predicted_vector = self.dogbreed_model.predict(bottleneck_feature)

        return self.dog_names[np.argmax(predicted_vector)]

