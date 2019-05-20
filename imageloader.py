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
import glob

class data_process:
  def __init__(self,dir):
    self.point = []
    self.label = []
    self.num_dict = {}
    self.dir = dir
    self.images = []
  def point_data_load(self):
    dir_list = []
    point_list = []
    label_list = []
    start_time = time.time()
    for dirpath, dirnames, filenames in os.walk(self.dir):
      dir_list.append(dirnames)
    
    if(dir_list[0] == []):
      for i in range(10):
        j = 0
        while(os.path.exists(self.dir + "/{}augs_{}.pickle".format(i,j))):
          j += 1
        for k in range(j):
          df = pd.read_pickle(self.dir + '/{}augs_{}.pickle'.format(i,k))
          points = df[['x','y']].to_numpy()
          labels = df[['label']].to_numpy()
          point_list.append(points)
          label_list.append(labels)

    else:
      for direct in dir_list[0]:
        folder = self.dir + '/' + direct
        for files in os.listdir(folder):
          if(files[-1] == 'v'):
            df = pd.read_csv(folder + '/' + files)
            df = df.loc[(df.x!=0) & (df.y !=0)]
            points = df[['x','y']].to_numpy()
            labels = df[['label']].to_numpy()
            point_list.append(points)
            label_list.append(labels)
    
    Point_DATA = np.array(point_list)
    Label_DATA = np.array(label_list)
    
    for i in range(np.size(Label_DATA,0)):
      Label_DATA[i] = np.unique(Label_DATA[i],axis=0)
    Label_DATA = Label_DATA.reshape((np.size(Label_DATA,0),1))
    unique, counts = np.unique(Label_DATA, return_counts=True)
    #print(unique)
    unique = ['0','1','2','3','4','5','6','7','8','9']
    num_dict = dict(zip(unique, counts))
    print("--- %s seconds ---" %(time.time() - start_time))

    self.point = Point_DATA
    self.label = Label_DATA
    self.num_dict = num_dict


  def image_make(self):

    for i in range(np.size(self.point,0)):
      img = np.zeros((800, 800, 1), np.uint8)
      for k in range(len(self.point[i])):
        if(k != len(self.point[i])-1):
          if(self.label[i][0][0][0] == 1):
            cv2.line(img, (self.point[i][k][0],self.point[i][k][1]), (self.point[i][k+1][0],self.point[i][k+1][1]), (255,255,255), 3)
          else:
            cv2.line(img, (self.point[i][k][0],self.point[i][k][1]), (self.point[i][k+1][0],self.point[i][k+1][1]), (255,255,255), 10)
      ret,thresh = cv2.threshold(img,127,255,0)
      contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)
      cnt = contours[0]
      x,y,w,h = cv2.boundingRect(cnt)
      if(x < 20 or y < 20):
        img = img[y:y+h+20,x:x+w+20]
      else:
        img = img[y-20:y+h+20,x-20:x+w+20]
      img = cv2.flip(img, 1)
      img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
      #print(self.label[i][0][0][0])
      j = 0
      while(os.path.exists('./DATA/image/{}img_{}.jpg'.format(self.label[i][0][0][0],j))):
        if(j < self.num_dict[str(self.label[i][0][0][0])]):
          j += 1
      cv2.imwrite('./DATA/image/{}img_{}.jpg'.format(self.label[i][0][0][0],j), img)

  def image_read(self):
    images = []
    labels = []
    for files in os.listdir('./DATA/image'):
      img = cv2.imread('./DATA/image/' + files, cv2.IMREAD_GRAYSCALE)
      images.append(img)
      labels.append(files[0])
    images = np.array(images)
    images = np.reshape(images,(-1,28,28,1))
    images = images/255
    self.images = images
    self.label = np.array(labels)
  
  def data_shuffle(self,point_only= False,image_only= False):
    if(point_only):
      indices = np.arange(self.point.shape[0])
      np.random.shuffle(indices)
      self.point = self.point[indices]
      self.label = self.label[indices]
    elif(image_only):
      indices = np.arange(self.images.shape[0])
      np.random.shuffle(indices)
      self.images = self.images[indices]
      self.label = self.label[indices]
    else:
      indices = np.arange(self.point.shape[0])
      np.random.shuffle(indices)
      self.point = self.point[indices]
      self.images = self.images[indices]
      self.label = self.label[indices]



