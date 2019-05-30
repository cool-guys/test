import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline 
from imageloader import data_process
import random
def DA_Jitter(X, sigma):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def GenerateRandomCurves(X, sigma=0.01, knot=7):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    #cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range)]).transpose()

def DA_MagWarp(X, sigma):
    return X * GenerateRandomCurves(X, sigma)

def DistortTimesteps(X, sigma=0.01):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    #tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum

def DA_TimeWarp(X, sigma=0.01):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    #X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    return X_new

def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)

def RandSampleTimesteps(X, nSample=80):
    X_new = np.zeros(X.shape)
    tt = np.zeros((nSample,X.shape[1]), dtype=int)
    tt[1:-1,0] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,1] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[-1,:] = X.shape[0]-1
    return tt

def DA_RandSampling(X, nSample=80):
    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros(X.shape)
    X_new[:,0] = np.interp(np.arange(X.shape[0]), tt[:,0], X[tt[:,0],0])
    X_new[:,1] = np.interp(np.arange(X.shape[0]), tt[:,1], X[tt[:,1],1])
    return X_new

def Rotation(X):
    theta = np.random.uniform(-1,1) * np.pi
    Q = np.array([[np.cos(theta/18),-np.sin(theta/18)],[np.sin(theta/18),np.cos(theta/18)]])
    point = []
    for i in range(50):
        point.append(np.matmul(Q,X[i]))
    
    return np.array(point)

        

dp = data_process('./DATA/test')
dp.point_data_load()
#dp.image_make()
#dp.image_read()



X_train = dp.point
Y_train =dp.label
print(int(Y_train[0]))
lnegth = np.size(X_train,0)
aug_list = []
for i in range(3000):
    a = random.randint(0,3)

    A = str(i//300)
    B = str(i%300)

    if(a == 0):
        point = DA_RandSampling(X_train[i%3000])
        C = A + ' ' + B + ' ' + 'rand'
    elif(a == 1):
        point = DA_TimeWarp(X_train[i%3000],0.05)
        C = A + ' ' + B + ' ' + 'TW'
    elif(a == 2):
        point = Rotation(X_train[i%3000])
        C = A + ' ' + B + ' ' + 'Rot'
    else:
        point = DA_Jitter(X_train[i%3000],4)
        C = A + ' ' + B + ' ' + 'jit'


    print(C)
    aug_list.append(C)
    dataframe = pd.DataFrame(point.astype(int), columns= ['x','y'])
    dataframe['label'] = int(Y_train[i%3000])
    if(i >= 3000 and i < 6000):
        dataframe.to_pickle("./DATA/aug/all/{}augs_{}.pickle".format((i-3000)//300,i%300 + 300))
    elif(i >= 6000 and i < 9000):
        dataframe.to_pickle("./DATA/aug/all/{}augs_{}.pickle".format((i-6000)//300,i%300 + 600))
    else:
        dataframe.to_pickle("./DATA/aug/all/{}augs_{}.pickle".format(i//300,i%300))

aug_list = np.array(aug_list)
df = pd.DataFrame(aug_list)
df.to_csv('aug_test',index=False)