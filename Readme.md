# Doggo App
<p align="center">
  <img src="data/space-doggo.jpg" style="width: 480px;">
  <br>
  CNN based dog breed classifier |
  <a href="https://doggo-app-195912.appspot.com/">DEMO</a>
</p>

## Overview

For Udacity's Artificial Intelligence Nanodegree, we built a [dog breed classifier](https://github.com/Rendiere/CNN-dog-breed-classifier) using deep Convolutional Neural Networks (CNNs) and [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) to utilize the current state of the art research. This model is based on the [ResNet50](https://arxiv.org/abs/1512.03385) architecture trained on the [ImageNet](http://www.image-net.org/) dataset, which at the time achieved state of the art results.


The purpose of this application is to serve predictions through a Flask API. A basic file upload page is served on the base route to test out the app.

**TODOs**
* Face recognition using Google Vision API
* Input data validation - protect against malicious uploads
* Unittests

## Getting Started

To run this model locally, follow these steps:

**Clone this repo**
```bash
git clone https://github.com/Rendiere/dog-app.git
cd dog-app
```

**Create a virtual environment**
```bash
conda create -n dog-app python=3
source activate dog-app
```


**Install requirements**
```bash
pip install -r requirements.txt
```

**Start the server**, running on  http://localhost:8080
```bash
python flask_api.py
```

## Things I learned

Here are a list of bugs I ran into and things I learned, for future reference.

### Bugs squashed

**Flask DEBUG and Keras aren't friends.**

When running a server locally, I usually set `debug=True` in the Flask `app.run()` command. However, Keras bugs out when you do this, failing to recognize the correct model.

**Google Apps Default Port**

The default port for applications running on the gcloud servers is `8080`. This is probably somewhere in the docs, but pretty hard to figure out if you've missed it, because the App Engine uses Nginx as a load balancer which returns a 502 error for any errors happening on your application.

So, setting `debug=False` (default), is necessary.

### Deploying to Google Cloud

I chose to deploy this app to Google Cloud's AppEngine. These are some points I wish I knew when I started.

**Standard vs Flexible**

The App Engine has two deployment flows: [Standard](https://cloud.google.com/appengine/docs/standard/python/) and [Flexible](https://cloud.google.com/appengine/docs/flexible/python/). There's a lot of overlap, but it seems like the Standard flow is outdated (and only supports python 2.7), so I chose the Flexible flow.

So, when using the documentation its _super_ important to know whether you are viewing the Standard or Flexible flow docs. They handle things vastly differently.

For example, deploying a python app with third-party libraries is a [complicated process](https://cloud.google.com/appengine/docs/standard/python/tools/using-libraries-python-27) using the Standard flow, but with the Flexible flow its as simple as including a `requirements.txt` in your root directory.
