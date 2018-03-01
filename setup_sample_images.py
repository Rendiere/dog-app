"""
For each dog breed, copy a random image
from the training set to be used as a
sample image.

Download the training dataset from Kaggle before running this script

Author: Renier Botha
Date: 1 March 2018
"""


import os
import random
from shutil import copyfile
import pandas as pd
from glob import glob

# Read in labels data
labels_df = pd.read_csv('data/labels.csv',index_col=0)

# read list of images
images_list = [os.path.basename(f).split('.')[0] for f in glob('data/train/*')]

# Create dataframe with images list
images_df = pd.DataFrame(images_list, columns=['id'])
images_df['breed'] = [labels_df.loc[i]['breed'] for i in images_list]

# For each dog breed, randomly select one image and copy
# to sample_images directory
for breed, breed_df in images_df.groupby('breed'):
    #     pick a random id
    ids = breed_df['id'].values
    chosen_id = random.choice(ids)

    chosen_fname = f'data/train/{chosen_id}.jpg'

    sample_fname = f'data/sample_images/{breed}.jpg'

    assert (os.path.isfile(chosen_fname))

    print(f'copying {chosen_id} to {sample_fname}')
    copyfile(chosen_fname, sample_fname)