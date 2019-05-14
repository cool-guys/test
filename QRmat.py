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


for i in range(10):
  
  while(os.path.exists("./DATA/Video/{}/{}train_{}".format(i,i,j))):
    j += 1

  for k in range(1000):
    df = pd.read_csv("./DATA/Video/{}/{}train_{}".format(i,i,k%15))
    df = df.loc[(df.x!=0) & (df.y !=0)]
    df_img = df[['x','y']].to_numpy()
    img = np.zeros((550, 550, 1), np.uint8)
    img_ = np.zeros((550, 550, 1), np.uint8)
    theta = np.random.normal(1.5,0.5) * np.pi
    a = np.random.normal(2.5,0.5)
    b = np.random.normal(2.5,0.5)
    c = np.random.normal(2.5,0.5)
    Q = np.array([[np.cos(theta/12),-np.sin(theta/12)],[np.sin(theta/12),np.cos(theta/12)]])
    R = np.array([[a*0.5,b*0.4],[0,c/a]])
    A = np.matmul(Q,R)
    #A = np.array([[1+a,0],[0,1+a]])
    for l in range(len(df_img)):
      if(k == 0):
        point_ = df_img[l]
      else:
        point_ = np.matmul(A,df_img[l])
        point_ = np.around(point_)
        point_ = point_.astype(int)
      if(point_[0] > 1000 or point_[1] > 1000 ):
        #print(A)
        print(i)
        print(k)
        print('asd',A)   
      augmented_points.append(point_)

    dataframe = pd.DataFrame(augmented_points, columns= ['x','y'])
    
    dataframe['label'] = i
    dataframe.to_csv("./DATA/test/{}augs_{}".format(i,k), index=False)
    augmented_images = []
    augmented_image = []
    augmented_point = []
    augmented_points = []

