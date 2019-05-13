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
rotation = 0

def do(image):
    # Get size before we rotate
    #print(image.shape)
    image = image.reshape((550,550))
    image = Image.fromarray(image.astype('uint8'),'L')
    x = image.size[0]
    y = image.size[1]

    # Rotate, while expanding the canvas size
    image = image.rotate(rotation, expand=True, resample=Image.BICUBIC)

    # Get size after rotation, which includes the empty space
    X = image.size[0]
    Y = image.size[1]

    # Get our two angles needed for the calculation of the largest area
    angle_a = abs(rotation)
    angle_b = 90 - angle_a

    # Python deals in radians so get our radians
    angle_a_rad = math.radians(angle_a)
    angle_b_rad = math.radians(angle_b)

    # Calculate the sins
    angle_a_sin = math.sin(angle_a_rad)
    angle_b_sin = math.sin(angle_b_rad)

    # Find the maximum area of the rectangle that could be cropped
    E = (math.sin(angle_a_rad)) / (math.sin(angle_b_rad)) * \
        (Y - X * (math.sin(angle_a_rad) / math.sin(angle_b_rad)))
    E = E / 1 - (math.sin(angle_a_rad) ** 2 / math.sin(angle_b_rad) ** 2)
    B = X - E
    A = (math.sin(angle_a_rad) / math.sin(angle_b_rad)) * B

    # Crop this area from the rotated image
    # image = image.crop((E, A, X - E, Y - A))
    image = image.crop((int(round(E)), int(round(A)), int(round(X - E)), int(round(Y - A))))

    # Return the image, re-sized to the size of the image passed originally
    return image.resize((x, y), resample=Image.BICUBIC)

for i in range(10):
  while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
    j += 1
    
  rotation = random.randint(-20,20)

  while(rotation == 0):
    rotation = random.randint(-20,20)

  for k in range(j):
    df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,k))
    df_img = df[['x','y']].to_numpy()
    img = np.zeros((550, 550, 1), np.uint8)
    img_ = np.zeros((550, 550, 1), np.uint8)

    for l in range(len(df_img)):
      img_[df_img[l][1]][df_img[l][0]] = 255
      img_set.append(img_)
      img_ = np.zeros((550, 550, 1), np.uint8)
    for image in img_set:
      augmented_images.append(do(image))
    #print(np.size(img_set,axis=0))
    for l in range(np.size(img_set,axis=0)):
      img_ = np.array(augmented_images[l])
      img_ = img_.reshape(-1)
      augmented_point.append(np.argmax(img_))
    for  l in range(len(augmented_point)):
      x = augmented_point[l]%550
      y = int(augmented_point[l]/550)
      augmented_points.append([x,y])
    dataframe = pd.DataFrame(augmented_points, columns= ['x','y'])
    
    dataframe['label'] = i
    dataframe.to_csv("./DATA/aug/{}aug_{}".format(i,k), index=False)
    augmented_images = []
    augmented_image = []
    augmented_point = []
    augmented_points = []
    img_set = []





'''
for i in range(np.size(X_DATA,0)):
  cv2.imwrite('DATA/test/test{}.jpg'.format(i), X_DATA[i])

for i in range(np.size(X_1DATA,0)):
  cv2.imwrite('DATA/test_1/test{}.jpg'.format(i), X_1DATA[i])
'''
'''
p = Augmentor.Pipeline()
p.status()
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.random_distortion
p.status()

batch_size = 128
g = p.keras_generator_from_array(X_1DATA, Y_DATA, batch_size=10)

X, y = next(g)

X = np.reshape(X,(-1,550,550))

print(X.shape)
print(np.max(X[0]))
print(X[0])
print(np.argsort(X[0],axis=0))
'''
'''
for i in range(np.size(X,0)):
  for j in range(np.max(X[i])-1):
    aug.append(np.where(X[i]==j+1))
  DP = pd.DataFrame(aug)
  DP.to_csv("./DATA/aug/aug_{}".format(i), index=False,columns=columns)
'''
#print(img_set.shape)



