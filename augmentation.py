import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline 
from imageloader import data_process
import random
import shutil

# https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data

#jitter: 데이터에 노이즈 추가, 랜덤한 10개의 index에 노이즈를 추가한다.
def DA_Jitter(X, sigma): 
    myNoise = np.random.normal(loc=0, scale=sigma, size=(1,2))
    for i in range(10):
        j = np.random.randint(0,X.shape[0])
        X[j] = X[j]+myNoise
    return X
# sigma는 표준편차
# 랜덤한 커브 만들기, knot은 그래프를 몇개로 나눌지에 대한 하이퍼파라미터, knot이 클수록 RandomCurves가 복잡해짐
# x축을 knot만큼 나누고 y의 값은 랜덤한 값을 가진 점을 만들고 Cubicspline을 써서 점들을 연결한다.
def GenerateRandomCurves(X, sigma=0.01, knot=3): 
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    #cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range)]).transpose()

# Randomcurves를 원래 데이터와 곱하여 특정구간의 크기를 늘리거나 줄인다.
def DA_MagWarp(X, sigma):
    return X * GenerateRandomCurves(X, sigma)

# tt_cum: random curve의 값을 점진적으로 더한다.
def DistortTimesteps(X, sigma=0.01):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    #tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum
# np.interp:보간법, x_range: 적용할범위, tt_new[:,0]: 기준이 되는범위(점진적으로 더해졌기에 특정범위는 길고 짧음이 있다), X[:,0]: y값의 범위
def DA_TimeWarp(X, sigma=0.01):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    #X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    return X_new
# 1~X.shape[0]에서 nSample-2만큼 뽑고 정렬하기
def RandSampleTimesteps(X, nSample=80):
    X_new = np.zeros(X.shape)
    tt = np.zeros((nSample,X.shape[1]), dtype=int)
    tt[1:-1,0] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,1] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[-1,:] = X.shape[0]-1
    return tt
# 점을 nSample-2만큼 뽑고 그 점을 이용하여 그래프를 추정하여 그린다.
def DA_RandSampling(X, nSample=100):
    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros(X.shape)
    X_new[:,0] = np.interp(np.arange(X.shape[0]), tt[:,0], X[tt[:,0],0])
    X_new[:,1] = np.interp(np.arange(X.shape[0]), tt[:,1], X[tt[:,1],1])
    return X_new

def Rotation(X):
    theta = np.random.uniform(-1,1) * np.pi
    Q = np.array([[np.cos(theta/18),-np.sin(theta/18)],[np.sin(theta/18),np.cos(theta/18)]])
    point = []

    for i in range(X.shape[0]):
        point.append(np.matmul(Q,X[i]))
    
    return np.array(point)

        

dp_train = data_process('./DATA/Centered/Alphabet/train',False)
dp_train.point_data_load()
#dp.image_make()
#dp.image_read()
#dp.data_shuffle(point_only=True)

dp_test = data_process('./DATA/Centered/Alphabet/test',False)
dp_test.point_data_load()
#dp.image_make()
#dp.image_read()
#dp.data_shuffle(point_only=True)
size = int(np.size(dp_train.point,0))
size_ = int(np.size(dp_test.point,0))

X_train = dp_train.point[:]
X_test = dp_test.point[:]

Y_train = dp_train.label[:]
Y_test = dp_test.label[:]

aug_list = []
if(dp_test.number):
    try:
        shutil.rmtree('./DATA/aug/all/train/number')
        os.mkdir('./DATA/aug/all/train/number')
        shutil.rmtree('./DATA/aug/all/test/number')
        os.mkdir('./DATA/aug/all/test/number')
        shutil.rmtree('./DATA/aug/all/train_org/number')
        os.mkdir('./DATA/aug/all/train_org/number')
    except OSError as e:
        if e.errno == 2:
            print('No such file or directory')
            pass
        else:
            raise

    for i in range(size * 1):
        a = random.randint(0,3)

        A = str(i//(size/10))
        B = str(i%(size/10))
        if(i > size):
            if(a == 0):
                point = DA_RandSampling(X_train[i%size])
                C = A + ' ' + B + ' ' + 'rand'
            elif(a == 1):
                point = DA_TimeWarp(X_train[i%size],0.01)
                C = A + ' ' + B + ' ' + 'TW'
            elif(a == 2):
                point = Rotation(X_train[i%size])
                C = A + ' ' + B + ' ' + 'Rot'
            else:
                point = DA_Jitter(X_train[i%size],5)
                C = A + ' ' + B + ' ' + 'jit'
        else:
            point = X_train[i%size]
            C = 'org'

        print(C)
        aug_list.append(C)
        dataframe = pd.DataFrame(point.astype(int), columns= ['x','y'])
        dataframe['label'] = int(Y_train[i%size])
        if(i >= size and i < size*2):
            dataframe.to_pickle("./DATA/aug/all/train/number/{}number{}.pickle".format(int((i-size)//(size/10)),int(i%(size/10) + (size/10))))
        elif(i >= size*2 and i < size*3):
            dataframe.to_pickle("./DATA/aug/all/train/number/{}number{}.pickle".format(int((i-size*2)//(size/10)),int(i%(size/10) + (size*2/10))))
        else:
            dataframe.to_pickle("./DATA/aug/all/train/number/{}number{}.pickle".format(int(i//(size/10)),int(i%(size/10))))
            dataframe.to_pickle("./DATA/aug/all/train_org/number/{}number{}.pickle".format(int(i//(size/10)),int(i%(size/10))))


    aug_list = np.array(aug_list)
    df = pd.DataFrame(aug_list)
    df.to_csv('aug_train',index=False)

    aug_list = []

    for i in range(size_):
        a = random.randint(0,3)

        A = str(i//(size_/10))
        B = str(i%(size_/10))
        if(i > size_):
            if(a == 0):
                point = DA_RandSampling(X_test[i%size_])
                C = A + ' ' + B + ' ' + 'rand'
            elif(a == 1):
                point = DA_TimeWarp(X_test[i%size_],0.03)
                C = A + ' ' + B + ' ' + 'TW'
            elif(a == 2):
                point = Rotation(X_test[i%size_])
                C = A + ' ' + B + ' ' + 'Rot'
            else:
                point = DA_Jitter(X_test[i%size_],5)
                C = A + ' ' + B + ' ' + 'jit'
        else:
            point = X_test[i%size_]
            C = 'org'

        print(C)
        aug_list.append(C)
        dataframe = pd.DataFrame(point.astype(int), columns= ['x','y'])
        dataframe['label'] = int(Y_test[i%size_])
        if(i >= size_ and i < size_*2):
            dataframe.to_pickle("./DATA/aug/all/test/number/{}number{}.pickle".format(int((i-size_)//(size_/10)),int(i%(size_/10) + (size_/10))))
        elif(i >= size_*2 and i < size_*3):
            dataframe.to_pickle("./DATA/aug/all/test/number/{}number{}.pickle".format(int((i-size_*2)//(size_/10)),int(i%(size_/10) + (size_*2/10))))
        else:
            dataframe.to_pickle("./DATA/aug/all/test/number/{}number{}.pickle".format(int(i//(size_/10)),int(i%(size_/10))))
else:

    try:
        shutil.rmtree('./DATA/aug/all/train/Alphabet')
        os.mkdir('./DATA/aug/all/train/Alphabet')
        shutil.rmtree('./DATA/aug/all/test/Alphabet')
        os.mkdir('./DATA/aug/all/test/Alphabet')
        shutil.rmtree('./DATA/aug/all/train_org/Alphabet')
        os.mkdir('./DATA/aug/all/train_org/Alphabet')
    except OSError as e:
        if e.errno == 2:
            print('No such file or directory')
            pass
        else:
            raise

    for i in range(size * 1):
        a = random.randint(0,3)

        A = str(i//(size/10))
        B = str(i%(size/10))
        if(i > size):
            if(a == 0):
                point = DA_RandSampling(X_train[i%size])
                C = A + ' ' + B + ' ' + 'rand'
            elif(a == 1):
                point = DA_TimeWarp(X_train[i%size],0.01)
                C = A + ' ' + B + ' ' + 'TW'
            elif(a == 2):
                point = Rotation(X_train[i%size])
                C = A + ' ' + B + ' ' + 'Rot'
            else:
                point = DA_Jitter(X_train[i%size],5)
                C = A + ' ' + B + ' ' + 'jit'
        else:
            point = X_train[i%size]
            C = 'org'

        print(C)
        aug_list.append(C)
        dataframe = pd.DataFrame(point.astype(int), columns= ['x','y'])
        dataframe['label'] = int(Y_train[i%size])
        if(i >= size and i < size*2):
            dataframe.to_pickle("./DATA/aug/all/train/Alphabet/{}_Alphabet{}.pickle".format(chr(97+int((i-size)//(size/26))),int(i%(size/26) + (size/26))))
        elif(i >= size*2 and i < size*3):
            dataframe.to_pickle("./DATA/aug/all/train/Alphabet/{}_Alphabet{}.pickle".format(chr(97+int((i-size*2)//(size/26))),int(i%(size/26) + (size*2/26))))
        else:
            dataframe.to_pickle("./DATA/aug/all/train/Alphabet/{}_Alphabet{}.pickle".format(chr(97+int(i//(size/26))),int(i%(size/26))))
            dataframe.to_pickle("./DATA/aug/all/train_org/Alphabet/{}_Alphabet{}.pickle".format(chr(97+int(i//(size/26))),int(i%(size/26))))


    aug_list = np.array(aug_list)
    df = pd.DataFrame(aug_list)
    df.to_csv('aug_train',index=False)

    aug_list = []

    for i in range(size_):
        a = random.randint(0,3)

        A = str(i//(size_/10))
        B = str(i%(size_/10))
        if(i > size_):
            if(a == 0):
                point = DA_RandSampling(X_test[i%size_])
                C = A + ' ' + B + ' ' + 'rand'
            elif(a == 1):
                point = DA_TimeWarp(X_test[i%size_],0.03)
                C = A + ' ' + B + ' ' + 'TW'
            elif(a == 2):
                point = Rotation(X_test[i%size_])
                C = A + ' ' + B + ' ' + 'Rot'
            else:
                point = DA_Jitter(X_test[i%size_],5)
                C = A + ' ' + B + ' ' + 'jit'
        else:
            point = X_test[i%size_]
            C = 'org'

        print(C)
        aug_list.append(C)
        dataframe = pd.DataFrame(point.astype(int), columns= ['x','y'])
        dataframe['label'] = int(Y_test[i%size_])
        if(i >= size_ and i < size_*2):
            dataframe.to_pickle("./DATA/aug/all/test/Alphabet/{}_Alphabet{}.pickle".format(chr(97+int((i-size_)//(size_/26))),int(i%(size_/26) + (size_/26))))
        elif(i >= size_*2 and i < size_*3):
            dataframe.to_pickle("./DATA/aug/all/test/Alphabet/{}_Alphabet{}.pickle".format(chr(97+int((i-size_*2)//(size_/26))),int(i%(size_/26) + (size_*2/26))))
        else:
            dataframe.to_pickle("./DATA/aug/all/test/Alphabet/{}_Alphabet{}.pickle".format(chr(97+int(i//(size_/26))),int(i%(size_/26))))    
    aug_list = np.array(aug_list)
    df = pd.DataFrame(aug_list)
    df.to_csv('aug_test',index=False)