import pandas as pd
import numpy as np
import os
import cv2

import random
import Augmentor
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import math
from math import floor, ceil
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

columns = ['x','y']
j = 0
x_data = []
x_1_data = []
y_data = []
aug = []
img_set = []
augmented_images = []
augmented_image = []
augmented_point = []
augmented_points = []

#taken from: https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(((-1,2)))
im_merge = np.array([242,522])
im_merge = im_merge.reshape((2,1,1))
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

print(im_merge_t.reshape((2)))


for i in range(10):
  
  while(os.path.exists("./DATA/Video/{}/{}train_{}".format(i,i,j))):
    j += 1
    
  rotation = random.randint(-20,20)

  while(rotation == 0):
    rotation = random.randint(-20,20)

  for k in range(j):
    df = pd.read_csv("./DATA/Video/{}/{}train_{}".format(i,i,k))
    df = df.loc[(df.x!=0) & (df.y !=0)]
    df_img = df[['x','y']].to_numpy()
    img = np.zeros((550, 550, 1), np.uint8)
    img_ = np.zeros((550, 550, 1), np.uint8)

    df_img = df_img.reshape((-1,2,1))
    augmented_points = elastic_transform(df_img, df_img.shape[1] * 2, df_img.shape[1] * 0.08, df_img.shape[1] * 0.08)

    dataframe = pd.DataFrame(augmented_points, columns= ['x','y'])
    
    dataframe['label'] = i
    dataframe.to_csv("./DATA/aug/{}augs_{}".format(i,k), index=False)
    augmented_images = []
    augmented_image = []
    augmented_point = []
    augmented_points = []
    img_set = []






