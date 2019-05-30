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
from scipy.ndimage import gaussian_filter
import time
columns = ['x','y']
j = 0
x_data = []
x_1_data = []
y_data = []
aug = []
point_list = []
augmented_images = []
augmented_image = []
augmented_point = []
augmented_points = []
rotation = 0
test = []

for i in range(10):
  j = 0
  while(os.path.exists("./DATA/Video/{}/{}train_{}".format(i,i,j))):
    j += 1
  #print(j)

  for k in range(300):
    start_time = time.time()
    df = pd.read_csv("./DATA/Video/{}/{}train_{}".format(i,i,k%j))
    df = df.loc[(df.x!=0) & (df.y !=0)]
    df_img = df[['x','y']].to_numpy()
    print(df_img.size/2)
    test.append(df_img.size/2)
    img = np.zeros((550, 550, 1), np.uint8)
    img_ = np.zeros((550, 550, 1), np.uint8)
    theta = np.random.uniform(-1,1) * np.pi

    a = np.random.uniform(1.3,2)*0.8
    b = np.random.uniform(-1.3,1.3)*0.6
    c = np.random.uniform(0.8,2)

    #while(abs(b) < 0.5):
      #b = np.random.uniform(-1.3,1.3)
    if(i == 1):
      a = np.random.uniform(1.3,2)*0.5
      b = np.random.uniform(-1.3,1.3)*0.5
      c = np.random.uniform(1,2)
    else:
      if(b < 0):
        while(a-b > 3.2*0.8):
          a = np.random.uniform(1.3,2)*0.8
          b = np.random.uniform(-1.3,1.3)*0.6
          while(c < 1.5):
            c = np.random.uniform(1,2)
      else:
        while(a-b < 1.5*0.8):
          a = np.random.uniform(1.3,2)*0.8
          b = np.random.uniform(-1.3,1.3)*0.6
          while(c < 1.5):
            c = np.random.uniform(1,2)



    Q = np.array([[np.cos(theta/18),-np.sin(theta/18)],[np.sin(theta/18),np.cos(theta/18)]])
    R = np.array([[a,b],[0,c]])
    A = np.matmul(Q,R)
    #A = np.array([[1+a,0],[0,1+a]])
    for l in range(len(df_img)):
      if(k == 0):
        point_ = df_img[l]
      else:
        point_ = np.matmul(Q,df_img[l])
        point_ = np.around(point_)
        point_ = point_.astype(int)
      if(point_[0] > 1000 or point_[1] > 1000 ):
        #print(A)
        print(i)
        print(k)
        print('asd',A)

      augmented_points.append(point_)

    df = pd.DataFrame(augmented_points, columns= ['x','y'])
    #df = df.loc[(df.x!=400) & (df.y !=400)]
    df_img_x = df[['x']].to_numpy()
    df_img_x_mean = np.mean(df_img_x)
    x_dif = int(275-df_img_x_mean)
    df_img_x += x_dif
    #print('x',df_img_x)
    df_img_y = df[['y']].to_numpy()
    df_img_y_mean = np.mean(df_img_y)
    y_dif = int(275-df_img_y_mean)
    df_img_y +=y_dif

    df_img = np.concatenate((df_img_x,df_img_y), axis=1)
    df_img = np.rint(df_img)
    df_img = df_img.astype(int)
    dataframe = pd.DataFrame(df_img, columns= ['x','y'])
    dataframe['label'] = i
    dataframe.to_pickle("./DATA/test/{}augs_{}.pickle".format(i,k))
    augmented_images = []
    augmented_image = []
    augmented_point = []
    augmented_points = []
    #print("--- %s seconds ---" %(time.time() - start_time))

test = np.array(test)
print(np.amax(test))
print(np.mean(test))
