from imageloader import data_process
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("./DATA/Video/4/4train_90")
df = df.loc[(df.x!=0) & (df.y !=0)]
df = df[['x','y']].to_numpy()

img = np.zeros((550, 550, 1), np.uint8)
for k in range(len(df)):
    if(k != len(df)-1):
        cv2.line(img, (df[k][0],df[k][1]), (df[k+1][0],df[k+1][1]), (255,255,255), 25)
    #img = cv2.flip(img, 1)
    img = cv2.resize(img,(550,550),interpolation=cv2.INTER_AREA)

#cv2.imshow('sgsg',img)
#cv2.waitKey(0)
df = pd.read_csv("./DATA/Video/4/4train_90")
df = df.loc[(df.x!=0) & (df.y !=0)]
df_img_x = df[['x']].to_numpy()
df_img_x_mean = np.mean(df_img_x)
x_dif = int(275-df_img_x_mean)
df_img_x += x_dif
df_img_y = df[['y']].to_numpy()
df_img_y_mean = np.mean(df_img_y)
y_dif = int(275-df_img_y_mean)
df_img_y +=y_dif
df_img = np.concatenate((df_img_x,df_img_y), axis=1)
df_img = np.rint(df_img)
df = df_img.astype(int)

img_2 = np.zeros((550, 550, 1), np.uint8)
for k in range(len(df)):
    if(k != len(df)-1):
        cv2.line(img_2, (df[k][0],df[k][1]), (df[k+1][0],df[k+1][1]), (255,255,255), 25)
    #img = cv2.flip(img, 1)
    img_2 = cv2.resize(img_2,(550,550),interpolation=cv2.INTER_AREA)

ret,thresh = cv2.threshold(img_2,10,255,0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
        img_3 = img_2[y-60:y+h+60,x-60:x+w+60]
    else:
        img_3 = img_2[y-a:y+h+a,x-a:x+w+a]
else:
    if(x < 40 or y < 40):
        img_3 = img_2[y:y+h+20,x:x+w+20]
    else:
        img_3 = img_2[y-40:y+h+40,x-40:x+w+40]

img_4 = cv2.resize(img_3,(28,28),interpolation=cv2.INTER_AREA)

fig, axes = plt.subplots(1, 4, figsize=(15, 4))

axes[0].imshow(img,aspect="auto",extent=[80,120,32,0],cmap=plt.cm.Blues)
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(img_2,aspect="auto",cmap=plt.cm.Blues)
axes[1].set_title('centered Image')
axes[1].axis('off')
axes[2].imshow(img_3,aspect="auto",cmap=plt.cm.Blues)
axes[2].set_title('ROI Image')
axes[2].imshow(img_3,aspect="auto",cmap=plt.cm.Blues)
axes[2].set_title('ROI Image')
axes[2].axis('off')
axes[3].imshow(img_4,aspect="auto",cmap=plt.cm.Blues)
axes[3].set_title('resized Image')
axes[3].axis('off')
'''
ax1.imshow(img)
ax1.set_title("Original Image")
ax2.imshow(img_2)
ax2.set_title("centered Image")
#ax3.imshow(img_3)
#ax3.set_title("ROI Image")
'''

plt.tight_layout() 
plt.show()
