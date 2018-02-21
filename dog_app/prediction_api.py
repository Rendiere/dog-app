import os
import io
import numpy as np

from PIL import Image
from flask import Flask, render_template, request

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.applications.resnet50 import ResNet50, preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/uploads'

IMAGE_SIZE = (224, 224)

Resnet50_model = None
dogbreed_model = None
features = None


# Load the dog names
fh = open('data/dogs_list.txt')
dog_names = [line[4:].rstrip().replace('_', ' ') for line in fh]
fh.close()

def load_model():
    # Load the full resnet model
    print('Loading ResNet model...', end='')
    global Resnet50_model
    Resnet50_model = ResNet50(weights='data/saved_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
    print('done')

    print('Loading the bottleneck features...', end='')
    global features
    features = np.load('data/bottleneck_features/DogResnet50Data.npz')
    print('done')

    # Load the model trained on dog breeds
    # TODO: Save the complete model and not only the weights
    # TODO: Add batch normalization
    print('Loading the dog breeds classifier...', end='')
    global dogbreed_model
    dogbreed_model = Sequential()
    dogbreed_model.add(GlobalAveragePooling2D(input_shape=features['train'].shape[1:]))
    dogbreed_model.add(Dropout(0.2))
    dogbreed_model.add(Dense(133, activation='softmax'))
    dogbreed_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    dogbreed_model.load_weights('data/saved_models/weights.best.ResNet50.hdf5')
    print('done')


def extract_Resnet50(tensor):
    return Resnet50_model.predict(preprocess_input(tensor))


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Load the file from request
    file = request.files['image']

    # Save the file to disk
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))

    # Make predictions
    predicted_vector = dogbreed_model.predict(bottleneck_feature)

    # return dog breed that is predicted by the model
    # TODO: Return Sample of predicted breed
    # TODO: Alter original image with bboxes and labels
    # TODO: Add probabilities
    return dog_names[np.argmax(predicted_vector)]


if __name__ == '__main__':
    print("Loading models, please wait while the server starts up...")
    load_model()

    print('\nSERVER UP AND RUNNING')
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)
