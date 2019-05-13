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
max_shear_left = 25
max_shear_right = 25

def do(image,direction):
    image = image.reshape((550,550))
    image = Image.fromarray(image.astype('uint8'),'L')
    width = image.size[0]
    height = image.size[1]
    # We use the angle phi in radians later
    phi = math.tan(math.radians(angle_to_shear))

    if direction == "x":
        # Here we need the unknown b, where a is
        # the height of the image and phi is the
        # angle we want to shear (our knowns):
        # b = tan(phi) * a
        shift_in_pixels = phi * height

        if shift_in_pixels > 0:
            shift_in_pixels = math.ceil(shift_in_pixels)
        else:
            shift_in_pixels = math.floor(shift_in_pixels)

        # For negative tilts, we reverse phi and set offset to 0
        # Also matrix offset differs from pixel shift for neg
        # but not for pos so we will copy this value in case
        # we need to change it
        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        # Note: PIL expects the inverse scale, so 1/scale_factor for example.
        transform_matrix = (1, phi, -matrix_offset,
                            0, 1, 0)

        image = image.transform((int(round(width + shift_in_pixels)), height),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC)

        #image = image.crop((abs(shift_in_pixels), 0, width, height))

        return image.resize((width, height), resample=Image.BICUBIC)

    elif direction == "y":
        shift_in_pixels = phi * width

        matrix_offset = shift_in_pixels
        if angle_to_shear <= 0:
            shift_in_pixels = abs(shift_in_pixels)
            matrix_offset = 0
            phi = abs(phi) * -1

        transform_matrix = (1, 0, 0,
                            phi, 1, -matrix_offset)

        image = image.transform((width, int(round(height + shift_in_pixels))),
                                Image.AFFINE,
                                transform_matrix,
                                Image.BICUBIC)

        #image = image.crop((0, abs(shift_in_pixels), width, height))

        return image.resize((width, height), resample=Image.BICUBIC)


for i in range(10):
  while(os.path.exists("./DATA/Video/{}/{}train_{}".format(i,i,j))):
    j += 1
    
  angle_to_shear = int(random.uniform((abs(max_shear_left)*-1) - 1, max_shear_right + 1))
  if angle_to_shear != -1: angle_to_shear += 1
  
  directions = ["x", "y"]
  direction = random.choice(directions)
    
  for k in range(j):
    df = pd.read_csv("./DATA/Video/{}/{}train_{}".format(i,i,k))
    df = df.loc[(df.x!=0) & (df.y !=0)]
    df_img = df[['x','y']].to_numpy()
    img = np.zeros((550, 550, 1), np.uint8)
    img_ = np.zeros((550, 550, 1), np.uint8)


    for l in range(len(df_img)):
      img_[df_img[l][1]][df_img[l][0]] = 255
      img_[df_img[l][1]+1][df_img[l][0]] = 255
      img_[df_img[l][1]-1][df_img[l][0]] = 255
      img_[df_img[l][1]][df_img[l][0]+1] = 255
      img_[df_img[l][1]][df_img[l][0]-1] = 255
      img_[df_img[l][1]+1][df_img[l][0]-1] = 255
      img_[df_img[l][1]-1][df_img[l][0]-1] = 255
      img_[df_img[l][1]+1][df_img[l][0]+1] = 255
      img_[df_img[l][1]-1][df_img[l][0]+1] = 255
      img_set.append(img_)
      img_ = np.zeros((550, 550, 1), np.uint8)
    for image in img_set:
      augmented_images.append(do(image,direction))
    #print(np.size(img_set,axis=0))
    for l in range(np.size(img_set,axis=0)):
      img_ = np.array(augmented_images[l])
      img_ = img_.reshape(-1)
      print(np.max(img_))
      augmented_point.append(np.argmax(img_))
    for  l in range(len(augmented_point)):
      x = augmented_point[l]%550
      y = int(augmented_point[l]/550)
      augmented_points.append([x,y])
    dataframe = pd.DataFrame(augmented_points, columns= ['x','y'])
    
    dataframe['label'] = i
    dataframe.to_csv("./DATA/aug/{}aug_s{}".format(i,k), index=False)
    augmented_images = []
    augmented_image = []
    augmented_point = []
    augmented_points = []
    img_set = []

