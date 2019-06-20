import pandas as pd
import numpy as np
import os
import cv2

import random
import matplotlib.pyplot as plt
import time

number = False


if(number == True):
    for i in range(10):
        j = 0
        start_time = time.time()
        while(os.path.exists("./DATA/Video/{}/{}train_{}".format(i,i,j))):
            j += 1

        for k in range(j):
            df = pd.read_csv("./DATA/Video/{}/{}train_{}".format(i,i,k))
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
            dataframe.to_pickle("./DATA/Centered/number/{}number{}.pickle".format(i,k))
else:
    for i in range(26):
        j = 0
        start_time = time.time()
        while(os.path.exists("./DATA/Video/{}/{}train_{}".format(chr(97+i),chr(97+i),j))):
            j += 1

        for k in range(j):
            df = pd.read_csv("./DATA/Video/{}/{}train_{}".format(chr(97+i),chr(97+i),k))
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
            dataframe.to_pickle("./DATA/Centered/Alphabet/{}_Alphabet{}.pickle".format(chr(97+i),k))    
print("--- %s seconds ---" %(time.time() - start_time))

