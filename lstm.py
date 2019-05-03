import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.preprocessing import MinMaxScaler

cb_checkpoint = ModelCheckpoint(filepath='model.hdf5',
                                verbose=1)

j = 0
img_dict = {}
img_dict_T = {}
data = []
x_test = []
data_ = []
for i in range(10):
  while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
    j += 1
  for k in range(j):
    df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,k))
    scaler = MinMaxScaler(feature_range=(0, 100))
    df = scaler.fit_transform(df)
    #df = df.astype(int)
    #df =df.values
    df.reshape((-1,2))
    data = np.array(list(data.append(df)))
    img_dict['{}'.format(i)]  = np.array(data)
  data = []
  df = []


DT = np.array(list(img_dict.values()))

DT =DT.reshape(10*j,-1)
for i in range(10*j):
  data_.append(DT[i][0])
DT = np.array(data_)
Y = []
for i in range(10):
    for _ in range(j):
        X.append(i)
Y = np.array(X)

Y = keras.utils.to_categorical(X,num_classes=10, dtype='float32')

def train_generator():
  n = 0
  while True:
    x_train = np.reshape(DT[n],(1,np.size(DT[n],0),2))
    y_train = []
    for i in range(np.size(DT[n],0)):
      y_train.append(X[i])
    y_train = np.reshape(y_train,(1,np.size(DT[n],0),10))
    y_train = X[n].reshape(1,10)
    #print(y_train)
    if(n < 109):
      n += 1
    else:
      n = 0
    yield x_train, y_train

'''
model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(None, 2)))
model.add(LSTM(32))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit_generator(train_generator() ,steps_per_epoch=100, epochs=30, verbose=1,callbacks=[cb_checkpoint])
'''

'''
new_model = keras.models.load_model('model.hdf5')
new_model.summary()
#loss, acc = new_model.evaluate(DT, X)
for i in range(10):
  x_test = np.reshape(DT[11*(i+1)-1],(1,np.size(DT[11*(i+1)-1],0),2))
  pred = new_model.predict(x_test)
  print(np.argmax(pred,axis=1))
#pred = new_model.predict(x_test)
#print(np.argmax(pred,axis=1))
#print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))
'''