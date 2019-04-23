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
  while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
    j += 1
  for j in range(j):
    df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,j))
    #scaler = MinMaxScaler(feature_range=(0, 100))
    #df = scaler.fit_transform(df)
    #df = df.astype(int)
    df =df.values
    
    img = np.zeros((500, 500, 1), np.uint8)
    for k in range(len(df)):
      if(k != len(df)-1):
        cv2.line(img, (df[k][0],df[k][1]), (df[k+1][0],df[k+1][1]), (255,255,255), 25)
    img = cv2.flip(img, 1)
    img = cv2.resize(img,(100,100),interpolation=cv2.INTER_AREA)
    img_data.append(img)
    img_dict['{}'.format(i)]  = np.array(img_data)
  img_data = []
  df = []
print(img_dict['0'][0].shape)
#cv2.imshow("draw", img_dict['0'][0])
#cv2.waitKey(0)



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
     
fig = plt.figure(n)

while(n<8):

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

