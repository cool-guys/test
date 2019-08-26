import pandas as pd
import numpy as np
import os
import cv2

import random
import matplotlib.pyplot as plt
import time
import shutil

number = False # True는 number False는 alphabet

##############################data를 중심으로 옮겨주고 확장자를 csv에서 pickle로 변경시킴##############################

if(number == True):
    try:
        shutil.rmtree('./DATA/Centered/number')
        os.mkdir('./DATA/Centered/number')
    except OSError as e:
        if e.errno == 2:
            print('No such file or directory')
            pass
        else:
            raise

    for i in range(10):
        j = 0
        start_time = time.time()
        while(os.path.exists("../Video/{}/{}train_{}".format(i,i,j))): #파일 개수 세기
            j += 1

        for k in range(j): #data 중심으로 옮기기
            df = pd.read_csv("../Video/{}/{}train_{}".format(i,i,k))
            df = df.loc[(df.x!=0) & (df.y !=0)]
            img = np.zeros((550, 550, 1), np.uint8)

            df_x = df[['x']].to_numpy()
            df_x_mean = np.mean(df_x)
            x_dif = int(275-df_x_mean)
            df_x += x_dif
            df_y = df[['y']].to_numpy()
            df_y_mean = np.mean(df_y)
            y_dif = int(275-df_y_mean)
            df_y +=y_dif

            df_img = np.concatenate((df_x,df_y), axis=1)
            df_img = np.rint(df_img)
            df_img = df_img.astype(int)

            dataframe = pd.DataFrame(df_img, columns= ['x','y'])
            dataframe['label'] = i
            dataframe.to_pickle("./DATA/Centered/number/{}number{}.pickle".format(i,k)) # pickle로 저장
else:
    try:
        shutil.rmtree('./DATA/Centered/Alphabet')
        os.mkdir('./DATA/Centered/Alphabet')
        os.mkdir('./DATA/Centered/Alphabet/train')
        os.mkdir('./DATA/Centered/Alphabet/test')
    except OSError as e:
        if e.errno == 2:
            print('No such file or directory')
            pass
        else:
            raise
    for i in range(26):
        j = 0
        start_time = time.time()
        while(os.path.exists("../Video/train_set/{}/{}train_{}".format(chr(97+i),chr(97+i),j))):
            j += 1

        for k in range(j):
            df = pd.read_csv("../Video/train_set/{}/{}train_{}".format(chr(97+i),chr(97+i),k))
            df = df.loc[(df.x!=0) & (df.y !=0)]
            if(df.empty):
                df = pd.read_csv("../Video/train_set/{}/{}train_{}".format(chr(97+i),chr(97+i),k))
            img = np.zeros((550, 550, 1), np.uint8)

            df_x = df[['x']].to_numpy()
            df_x_mean = np.mean(df_x)
            x_dif = int(275-df_x_mean)
            print(i,k)
            df_x += x_dif
            df_y = df[['y']].to_numpy()
            df_y_mean = np.mean(df_y)
            y_dif = int(275-df_y_mean)
            df_y +=y_dif

            df_img = np.concatenate((df_x,df_y), axis=1)
            df_img = np.rint(df_img)
            df_img = df_img.astype(int)

            dataframe = pd.DataFrame(df_img, columns= ['x','y'])
            dataframe['label'] = i
            dataframe.to_pickle("./DATA/Centered/Alphabet/train/{}_Alphabet{}.pickle".format(chr(97+i),k))    

    for i in range(26):
        j = 0
        start_time = time.time()
        while(os.path.exists("../Video/test_set/{}/{}train_{}".format(chr(97+i),chr(97+i),j))):
            j += 1

        for k in range(j):
            df = pd.read_csv("../Video/test_set/{}/{}train_{}".format(chr(97+i),chr(97+i),k))
            df = df.loc[(df.x!=0) & (df.y !=0)]
            if(df.empty):
                df = pd.read_csv("../Video/test_set/{}/{}train_{}".format(chr(97+i),chr(97+i),k))
            img = np.zeros((550, 550, 1), np.uint8)

            df_x = df[['x']].to_numpy()
            df_x_mean = np.mean(df_x)
            x_dif = int(275-df_x_mean)
            print(i,k)
            df_x += x_dif
            df_y = df[['y']].to_numpy()
            df_y_mean = np.mean(df_y)
            y_dif = int(275-df_y_mean)
            df_y +=y_dif

            df_img = np.concatenate((df_x,df_y), axis=1)
            df_img = np.rint(df_img)
            df_img = df_img.astype(int)

            dataframe = pd.DataFrame(df_img, columns= ['x','y'])
            dataframe['label'] = i
            dataframe.to_pickle("./DATA/Centered/Alphabet/test/{}_Alphabet{}.pickle".format(chr(97+i),k))      

