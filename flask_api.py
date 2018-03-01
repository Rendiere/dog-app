import io
import os
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

from model import CNNModel

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    global model

    # Load the file from request
    file = request.files['image']

    file_bytes = io.BytesIO(file.read())

    image = Image.open(file_bytes)

    prediction = model.predict(image)

    print(prediction)

    sample_file = f'data/sample-images/{prediction}.jpg'

    print(sample_file)

    assert(os.path.isfile(sample_file))

    return send_file(sample_file, mimetype='image')


if __name__ == '__main__':
    print("Loading models, please wait while the server starts up...")
    model = CNNModel()

    print('\nSERVER UP AND RUNNING')
    app.run(host='0.0.0.0', port=8080, debug=False, )
