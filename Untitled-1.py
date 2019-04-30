import pandas as pd
import numpy as np
import os
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import keras

cb_checkpoint = ModelCheckpoint(filepath='model.hdf5',
                                verbose=1)

j = 0
img_dict = {}
data = []
data_ = []
for i in range(10):
  while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
    j += 1
  for k in range(j):
    df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,k))
    #scaler = MinMaxScaler(feature_range=(0, 100))
    #df = scaler.fit_transform(df)
    #df = df.astype(int)
    df =df.values
    #print(df.shape)    
    df.reshape((-1,2))
    data.append(df)
    img_dict['{}'.format(i)]  = np.array(data)
  data = []
  df = []


DT = np.array(list(img_dict.values()))

DT =DT.reshape(100,-1)
for i in range(100):
  data_.append(DT[i][0])
DT = np.array(data_)

X = []
for i in range(10):
    for j in range(10):
        X.append(i)
X = np.array(X)
print(X[0])
X = keras.utils.to_categorical(X,num_classes=10, dtype='float32')
#print(X)

def train_generator():
  n = 0
  while True:
    x_train = np.reshape(DT[n],(1,np.size(DT[n],0),2))
    #y_train = []
    #for i in range(np.size(DT[n],0)):
    #  y_train.append(X[i])
    #y_train = np.reshape(y_train,(1,np.size(DT[n],0),10))
    #print(x_train)
    #y_train = np.transpose(X[n])
    y_train = X[n].reshape(1,10)
    if(n < 99):
      n += 1
    else:
      n = 0
    yield x_train, y_train

model = keras.models.Sequential()
model.add(keras.layers.LSTM(128, input_shape=(None, 2),return_sequences=True))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.LSTM(64))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()


model.fit_generator(train_generator() ,steps_per_epoch=80, epochs=100, verbose=1,callbacks=[cb_checkpoint])
#pred = model.predict(np.reshape(DT[n],(1,np.size(DT[5],0),2)))
#print(np.argmax(pred,axis=1))

'''
new_model = keras.models.load_model('model.hdf5')
new_model.summary()
#loss, acc = new_model.evaluate(DT, X)
x_test = np.reshape(DT[15],(1,np.size(DT[15],0),2))
pred = new_model.predict(x_test)
print(np.argmax(pred,axis=1))
#print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))
'''