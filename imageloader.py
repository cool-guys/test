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
import shutil

class data_process:
  def __init__(self,dir,number):
    self.point = []
    self.label = []
    self.dir = dir
    self.images = []
    self.seq_length = []
    self.number = number
    self.class_num ={}
    if(self.number == True):
      for i in range(10):
        self.class_num[chr(i)] = 0
    else:
      for i in range(26):
        self.class_num[chr(97+i)] = 0

  def point_data_load(self):
    dir_list = []
    point_list = []
    label_list = []
    length_list = []
    start_time = time.time()

    for dirpath, dirnames, filenames in os.walk(self.dir):
      dir_list.append(dirnames)
    if(self.number):
      for i in range(10):
        j = 0
        while(os.path.exists(self.dir + "/{}number{}.pickle".format(i,j))):
          j += 1
        for k in range(j):
          df = pd.read_pickle(self.dir + '/{}number{}.pickle'.format(i,k))
          points = df[['x','y']].to_numpy()
          labels = int(df[['label']].to_numpy()[0,0])

          for l in range(26):
            if(l == labels):
              self.class_num[chr(97+l)] += 1
          point_list.append(points)
          label_list.append(labels)


    else:
      for i in range(26):
        j = 0
        while(os.path.exists(self.dir + "/{}_Alphabet{}.pickle".format(chr(97+i),j))):
          j += 1
        for k in range(j):
          df = pd.read_pickle(self.dir + '/{}_Alphabet{}.pickle'.format(chr(97+i),k))
          points = df[['x','y']].to_numpy()

          labels = int(df[['label']].to_numpy()[0,0])

          for l in range(26):
            if(l == labels):
              self.class_num[chr(97+l)] += 1
          point_list.append(points)
          label_list.append(labels)

  
    Point_DATA = np.array(point_list)
    Label_DATA = np.array(label_list)

    #for i in range(np.size(Label_DATA,0)):
    #  Label_DATA[i] = np.unique(Label_DATA[i],axis=0)
    Label_DATA = Label_DATA.reshape((np.size(Label_DATA,0),1))


    self.point = Point_DATA
    self.label = Label_DATA

  def image_make(self):

    if(self.dir == './DATA/aug/all/train/number'):
      try:
          shutil.rmtree('./DATA/image/train/number')
          os.mkdir('./DATA/image/train/number')
          shutil.rmtree('./DATA/image/test/number')
          os.mkdir('./DATA/image/test/number')
      except OSError as e:
          if e.errno == 2:
              print('No such file or directory')
              pass
          else:
              raise
    elif(self.dir == './DATA/aug/all/train/Alphabet'):     
      try:
          shutil.rmtree('./DATA/image/train/Alphabet')
          os.mkdir('./DATA/image/train/Alphabet')
          shutil.rmtree('./DATA/image/test/Alphabet')
          os.mkdir('./DATA/image/test/Alphabet')
      except OSError as e:
          if e.errno == 2:
              print('No such file or directory')
              pass
          else:
              raise  
    else:
      pass
    for i in range(np.size(self.point,0)):
      img = np.zeros((550, 550, 1), np.uint8)
#      if(self.point[i][49][0] != 0):
      for k in range(len(self.point[i])):
        if(k != len(self.point[i])-1):
          cv2.line(img, (self.point[i][k][0],self.point[i][k][1]), (self.point[i][k+1][0],self.point[i][k+1][1]), (255,255,255), 20)
        
      ret,thresh = cv2.threshold(img,10,255,0)
      contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      
      if(len(contours) != 0):
        cnt = contours[0]
        if(len(contours) > 1):
          x_ = []
          y_ = []
          xw = []
          yh = []
          for l in range(len(contours)):
            x,y,w,h = cv2.boundingRect(contours[l])
            x_.append(x)
            y_.append(y)
            xw.append(x+w)
            yh.append(y+h)
          x = min(x_)
          y = min(y_)
          w = max(xw) - x
          h = max(yh) - y
        else:    
          x,y,w,h = cv2.boundingRect(cnt)
        a = 0
        while(x > a and y > a and x + w > a and y + h > a):
          a += 1 
        #if(x < 40 or y < 40):
          #img = img[y:y+h+20,x:x+w+20]
        #else:
        if(a != 0):
          if(a > 60):
            img = img[y-60:y+h+60,x-60:x+w+60]
          else:
            img = img[y-a:y+h+a,x-a:x+w+a]
        else:
          if(x < 40 or y < 40):
            img = img[y:y+h+20,x:x+w+20]
          else:
            img = img[y-40:y+h+40,x-40:x+w+40]
      
      img = cv2.flip(img, 1)
      img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
      j = 0
      
      if(self.number):      

        if(self.dir == './DATA/aug/all/train/number'):     
          if(i == 0):     
            cv2.imwrite('./DATA/image/train/number/{}img_{}.jpg'.format(self.label[i,0],j), img)
          else:
            while(os.path.exists('./DATA/image/train/{}img_{}.jpg'.format(self.label[i,0],j))):
              j += 1

            cv2.imwrite('./DATA/image/train/number/{}img_{}.jpg'.format(self.label[i,0],j), img)

        elif(self.dir == './DATA/aug/all/test/number'):
          if(i == 0):        
            cv2.imwrite('./DATA/image/test/number/{}img_{}.jpg'.format(self.label[i,0],j), img)
          else:
            while(os.path.exists('./DATA/image/test/{}img_{}.jpg'.format(self.label[i,0],j))):
              j += 1

            cv2.imwrite('./DATA/image/test/number/{}img_{}.jpg'.format(self.label[i,0],j), img)
        else: 
          if(i == 0):          
            cv2.imwrite('./DATA/image/original/number/{}img_{}.jpg'.format(self.label[i,0],j), img)
          else:
            while(os.path.exists('./DATA/image/original/number/{}img_{}.jpg'.format(self.label[i,0],j))):       
              j += 1  
            cv2.imwrite('./DATA/image/original/number/{}img_{}.jpg'.format(self.label[i,0],j), img)

      else:
        if(self.dir == './DATA/aug/all/train/Alphabet'):
          if(i == 0):
            cv2.imwrite('./DATA/image/train/Alphabet/{}_img_{}.jpg'.format(chr(97+int(self.label[i,0])),j), img)

          else:
            while(os.path.exists('./DATA/image/train/Alphabet/{}_img_{}.jpg'.format(chr(97+int(self.label[i,0])),j))):
              j += 1
            cv2.imwrite('./DATA/image/train/Alphabet/{}_img_{}.jpg'.format(chr(97+int(self.label[i,0])),j), img)
                
        elif(self.dir == './DATA/aug/all/test/Alphabet'):
          if(i == 0):
            cv2.imwrite('./DATA/image/test/Alphabet/{}_img_{}.jpg'.format(chr(97+int(self.label[i,0])),j), img)

          else:
            while(os.path.exists('./DATA/image/test/Alphabet/{}_img_{}.jpg'.format(chr(97+int(self.label[i,0])),j))):
              j += 1
            cv2.imwrite('./DATA/image/test/Alphabet/{}_img_{}.jpg'.format(chr(97+int(self.label[i,0])),j), img)
                
        else:
          if(i == 0):
            cv2.imwrite('./DATA/image/original/Alphabet/{}_img_{}.jpg'.format(chr(97+int(self.label[i,0]),j)), img)

          else:
            while(os.path.exists('./DATA/image/original/Alphabet/{}_img_{}.jpg'.format(chr(97+int(self.label[i,0])),j))):
              j += 1
            cv2.imwrite('./DATA/image/original/Alphabet/{}_img_{}.jpg'.format(chr(97+int(self.label[i,0]),j)), img)    

      
      img = np.array(img)
      img = np.reshape(img,(28,28,1))
      img = img/255
      self.images.append(img)
    

    self.images = np.array(self.images)

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

  def sequence_64(self):
    point_list = []
    
    for i in range(np.size(self.point,0)):
      points = self.point[i]
      leng = np.size(points,0)
      point = []
      if(leng > 64):
        for l in range(64):
          point.append(points[int(leng*l/64)])
      else:
        for l in range(leng):
          point.append(points[l])#points[l]
        while(len(point) != 64):
          point.append(points[leng-1])#points[leng-1]
      point = np.array(point)
      point.reshape((64,2))

      for j in range(62):
        grads = np.array([point[j+2][0] - point[j][0],point[j+2][1] - point[j][1]])
        if(abs(point[j][0] - point[j+1][0]) > 100 or abs(point[j][1] - point[j+1][1]) > 100):
          point[j+1][0] = int(point[j][0] + grads[0]/2)
          point[j+1][1] = int(point[j][1] + grads[1]/2)
        else:
          pass
      point_list.append(point)
  #  for i in range(np.size(self.point,0)):

    Point_DATA = np.array(point_list)
    Point_DATA = Point_DATA.reshape((-1,64,2))
    self.point = Point_DATA


'''
      else:
        for k in range(len(self.point[i]) -1):
          if(k != len(self.point[i])-1 and self.point[i][k][0] != 0 and self.point[i][k+1][0] != 0):
            cv2.line(img, (self.point[i][k][0],self.point[i][k][1]), (self.point[i][k+1][0],self.point[i][k+1][1]), (255,255,255), 20)
'''      