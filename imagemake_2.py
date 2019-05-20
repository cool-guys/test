import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler,StandardScaler,QuantileTransformer
import os

j = 0
n = 0
img_data = []
img_dict = {}
img = np.zeros((100, 100, 1), np.uint8)

#368x496
for i in range(10):
  j = 0
  while(os.path.exists("./DATA/Video/{}/{}train_{}".format(i,i,j))):
    j += 1
  for j in range(j):
    df = pd.read_csv("./DATA/Video/{}/{}train_{}".format(i,i,j))
    df = df.loc[(df.x!=0) & (df.y !=0)]

    #scaler = MinMaxScaler(feature_range=(0, 100))
    #df = scaler.fit_transform(df)
    #df = df.astype(int)

    df =df.values
    
    img = np.zeros((500, 500, 1), np.uint8)
    for k in range(len(df)):
      if(k != len(df)-1):
        cv2.line(img, (df[k][0],df[k][1]), (df[k+1][0],df[k+1][1]), (255,255,255), 25)
    img = cv2.flip(img, 1)
    img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
    img_data.append(img)
    img_dict['{}'.format(i)]  = np.array(img_data)
  img_data = []
  df = []
print(img_dict['0'][0].shape)
#cv2.imshow("draw", img_dict['0'][0])
#cv2.waitKey(0)


max_val = 7
def onclick(event):
    global n
    if(event.key == 'right'):
      n += 1
      print("right")
      plt.close()

    elif(event.key == 'left'):
      n -= 1
      print("left")
      plt.close()
    elif(event.key == 'enter'):
      exit()


    elif(event.key =='1'):
      dir = './DATA/1/'
      files = os.listdir(dir)
      k=1
      idx=n
      print('figure {} , number {} is deleted'.format(n,k))
      for filename in files:
        while(idx < max_val):
                os.rename(dir + "{}train_{}".format(k,idx+1), dir +"{}train_{}".format(k,idx))
                idx=idx+1


    elif(event.key =='2'):
      dir = './DATA/2/'
      files = os.listdir(dir)
      idx=n
      k=2
      print('figure {} , number {} is deleted'.format(n,k))
      for filename in files:
        while(idx < max_val):
                os.rename(dir + "{}train_{}".format(k,idx+1), dir +"{}train_{}".format(k,idx))
                idx=idx+1
     
    elif(event.key =='3'):
      dir = './DATA/3/'
      files = os.listdir(dir)
      idx=n
      k=3
      print('figure {} , number {} is deleted'.format(n,k))
      for filename in files:
        while(idx < max_val):
                os.rename(dir + "{}train_{}".format(k,idx+1), dir +"{}train_{}".format(k,idx))
                idx=idx+1

    elif(event.key =='4'):
      dir = './DATA/4/'
      files = os.listdir(dir)
      idx=n
      k=4
      print('figure {} , number {} is deleted'.format(n,k))
      for filename in files:
        while(idx < max_val):
                os.rename(dir + "{}train_{}".format(k,idx+1), dir +"{}train_{}".format(k,idx))
                idx=idx+1

    elif(event.key =='5'):
      dir = './DATA/5/'
      files = os.listdir(dir)
      idx=n
      k=5
      print('figure {} , number {} is deleted'.format(n,k))
      for filename in files:
        while(idx < max_val):
                os.rename(dir + "{}train_{}".format(k,idx+1), dir +"{}train_{}".format(k,idx))
                idx=idx+1

    elif(event.key =='6'):
      dir = './DATA/6/'
      files = os.listdir(dir)
      idx=n
      k=6
      print('figure {} , number {} is deleted'.format(n,k))
      for filename in files:
        while(idx < max_val):
                os.rename(dir + "{}train_{}".format(k,idx+1), dir +"{}train_{}".format(k,idx))
                idx=idx+1

    elif(event.key =='7'):
      dir = './DATA/7/'
      files = os.listdir(dir)
      idx=n
      k=7
      print('figure {} , number {} is deleted'.format(n,k))
      for filename in files:
        while(idx < max_val):
                os.rename(dir + "{}train_{}".format(k,idx+1), dir +"{}train_{}".format(k,idx))
                idx=idx+1

    elif(event.key =='8'):
      dir = './DATA/8/'
      files = os.listdir(dir)
      idx=n
      k=8
      print('figure {} , number {} is deleted'.format(n,k))
      for filename in files:
        while(idx < max_val):
                os.rename(dir + "{}train_{}".format(k,idx+1), dir +"{}train_{}".format(k,idx))
                idx=idx+1

    elif(event.key =='9'):
      dir = './DATA/9/'
      files = os.listdir(dir)
      idx=n
      k=9
      print('figure {} , number {} is deleted'.format(n,k))
      for filename in files:
        while(idx < max_val):
                os.rename(dir + "{}train_{}".format(k,idx+1), dir +"{}train_{}".format(k,idx))
                idx=idx+1  

    elif(event.key =='0'):
      dir = './DATA/0/'
      files = os.listdir(dir)
      idx=n
      print('figure {} , number {} is deleted'.format(n,k))
      k=0
      for filename in files:
        while(idx < max_val):
                os.rename(dir + "{}train_{}".format(k,idx+1), dir +"{}train_{}".format(k,idx))
                idx=idx+1
     
fig = plt.figure(n)

while(1):

  fig = plt.figure(n)
  for i in range(10):
    subplot = fig.add_subplot(2, 5, i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])

    subplot.set_title('{}'.format(i))
    #cid = plt.gcf().canvas.mpl_connect('key_press_event', onclick)
    subplot.imshow(img_dict['{}'.format(i)][n],cmap=plt.cm.gray_r)
   
  
  cid = plt.gcf().canvas.mpl_connect('key_press_event', onclick)
  plt.show(fig)
  


  #cid = plt.gcf().canvas.mpl_connect('key_press_event', onclick)
 # plt.close(fig)


#print(img_dict['1'][3].shape)

