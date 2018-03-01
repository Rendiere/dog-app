"""
For each dog breed, copy a random image
from the training set to be used as a
sample image.

Download the testing dataset from Udacity before running this script

Author: Renier Botha
Date: 1 March 2018
"""

import os
import random
from glob import glob
from shutil import copyfile

# List of directories with images
dirs = glob('data/test/*')

for d in dirs:
    # Get just the dog name
    dog_name = d[14:]

    # Get a list of images
    images_list = glob(f'{d}/*.jpg')

    # Pick a random image
    rand_image = random.choice(images_list)

    # Create the filename for the sample image
    sample_fname = f'data/sample_images/{dog_name}.jpg'

    print('Copying random image for ',dog_name)

    # Copy the image
    copyfile(rand_image, sample_fname)