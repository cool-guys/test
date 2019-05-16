import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, BatchNormalization,Activation, ZeroPadding2D,CuDNNLSTM, Bidirectional,LSTM,Input,concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
import tensorflow as tf

from keras.callbacks import ModelCheckpoint,EarlyStopping

cb_checkpoint = ModelCheckpoint(filepath='model.hdf5',save_best_only=True,
                                verbose=1)

columns = ['x','y', 'label']
j = 0
x_data = []
y_data = []
scaler = MinMaxScaler((0,400))

for i in range(10):
  while(os.path.exists("./DATA/test/{}augs_{}".format(i,j))):
    j += 1
  for k in range(j):
    df = pd.read_pickle("./DATA/test/{}augs_{}".format(i,k))
    df_img = df[['x','y']].to_numpy()
    img = np.zeros((800, 800, 1), np.uint8)
    for k in range(len(df_img)):
      if(k != len(df_img)-1):
        cv2.line(img, (df_img[k][0],df_img[k][1]), (df_img[k+1][0],df_img[k+1][1]), (255,255,255), 30)
    img = cv2.flip(img, 1)
    img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)


    x_data.append(img)
    y_data.append(df[['x','y']].to_numpy())

    
X_DATA = np.array(x_data)
X_DATA = np.reshape(X_DATA,(-1,28,28,1))
X_DATA = X_DATA/255
Y_DATA = np.array(y_data)

#print(Y_DATA)




X_train, X_test, Y_train, Y_test = train_test_split(X_DATA, Y_DATA, test_size=0.1, random_state=32)



X_train = X_train
#DX_val = np.reshape(X_val,(-1,28*28))
DX_train = np.reshape(X_train,(-1,28*28))
DX_test = np.reshape(X_test,(-1,28*28))
Y_train_ = []
Y_test_ = []
'''
for i in range(np.size(Y_train,0)):
  len_size = np.size(Y_train[i],0)
  if(len_size <= 100):
    for j in range(100 -len_size):
      #print(Y_train[i].shape)
      last = Y_train[i][len_size-1].reshape((-1,2))
      Y_train[i] = np.append(Y_train[i],last,axis=0)
    Y_train_.append(Y_train[i])
  else:
    len_size - 100

Y_train = np.array(Y_train_)
Y_train = Y_train.reshape((-1,100,2))  

for i in range(np.size(Y_test,0)):
  len_size = np.size(Y_test[i],0)
  if(len_size <= 100):
    for j in range(100 -len_size):
      last_ = Y_test[i][len_size-1].reshape((1,2))
      Y_test[i] = np.append(Y_test[i],last_,axis=0)
    Y_test_.append(Y_test[i])
Y_test = np.array(Y_test_)
Y_test = Y_test.reshape((-1,100,2))  
'''
Y_train = keras.preprocessing.sequence.pad_sequences(Y_train, maxlen=100, padding='post', dtype='float32')
Y_test = keras.preprocessing.sequence.pad_sequences(Y_test, maxlen=100, padding='post', dtype='float32')

'''
for i in range(809):
  lens = np.size(Y_train_[i+90],0)
  if(100 - lens) > 0:
    for j in range(100 - lens):
      Y_train[i][j+lens-1] = Y_train_[i+90][lens-1]

for i in range(100):
  lens = np.size(Y_test[i],0)
  if(100 - lens) > 0:
    for j in range(100 - lens):
      Y_test[i][j+lens-1] = Y_test_[i][lens-1]
'''
#print('asdsd',np.where(Y_train==0))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def custom_activation(x):
  sig = tf.math.sigmoid(x) * 100
  tf.cast(sig,tf.int32)
  return sig
def my_cross_entropy(y_true, y_pred):
  dif_p = []
  dif_t = []

  for i in range(1,2):
    xp_d = tf.math.subtract(y_pred[:,i-1,0],y_pred[:,i,0])
    yp_d = tf.math.subtract(y_pred[:,i-1,1],y_pred[:,i,1])
    xt_d = tf.math.subtract(y_true[:,i-1,0],y_true[:,i,0])
    yt_d = tf.math.subtract(y_true[:,i-1,1],y_true[:,i,1])
    dif_p.append(tf.math.truediv(xp_d,yp_d))
    dif_t.append(tf.math.truediv(xt_d,yt_d))
  if
  dif_p = tf.stack(dif_p)
  dif_t = tf.stack(dif_t)

  #keras.losses.mean_squared_error(dif_t, dif_p)
  print('asgasgasgsgsg',y_pred[0])
  
  return keras.losses.mean_squared_error(tf.math.truediv(xt_d,yt_d), tf.math.truediv(xp_d,yp_d))
'''
input_1 = Input(shape=(28, 28, 1))

x_1 = Conv2D(40, kernel_size=5, padding="same", activation = 'relu')(input_1)
x_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_1)
x_1 = Dropout(0.5)(x_1)
x_1 = Conv2D(70, kernel_size=3, padding="same", activation = 'relu')(x_1)
x_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x_1)
x_1 = Dropout(0.5)(x_1)
x_1 = Conv2D(200, kernel_size=3, padding="same", activation = 'relu')(x_1)
x_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x_1)
x_1 = Dropout(0.5)(x_1)

x_1 = Conv2D(512, kernel_size=3, padding="valid", activation = 'relu')(x_1)
x_1 = Dropout(0.5)(x_1)
x_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x_1)
x_1 = Flatten()(x_1)
x_1 = Dense(units=128, activation='relu')(x_1)
x_1 = Dropout(0.5)(x_1)

input_2 = Input(shape=(28*28,))

x_2 = Dense(512, activation='relu')(input_2)
x_2 = Dense(512,activation='relu')(x_2)
x_2 = Dense(256, activation='relu')(x_2)
x_2 = Dense(256, activation='relu')(x_2)

merged = concatenate([x_1,x_2])
m = Dense(500, activation='relu')(merged)
m = Reshape((100,-1))(m)
#m = CuDNNLSTM(256, input_shape=(25,2), return_sequences=True)(m)
#m = CuDNNLSTM(128, return_sequences=True)(m)
m = Dense(256, activation='relu')(m)
m = Dense(2)(m)
'''
'''
model = Sequential()
model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=keras.regularizers.l2(0.01),
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3),kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3),kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))


model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3),kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3),kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))



model.add(Flatten())
model.add(Dense(400))
model.add(Dropout(0.5))
model.add(Reshape((100,-1)))
model.add(CuDNNLSTM(256, input_shape=(100,2), return_sequences=True))
model.add(Dense(128))
model.add(Dense(2))
'''

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28*28,)))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Reshape((100,-1)))
model.add(CuDNNLSTM(256, input_shape=(100,2), return_sequences=True))
model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dense(128))
model.add(Dense(2, activation='relu'))

#model = Model(inputs=[input_1, input_2], outputs = m)
model.summary()

early_stopping = EarlyStopping(patience = 200)

model.compile(loss=my_cross_entropy, optimizer=keras.optimizers.Adam(lr=0.005,decay=1e-5), metrics=['accuracy'])
hist = model.fit(DX_train, Y_train,
                 batch_size=64,
                 epochs=1000,
                 validation_data=(DX_test, Y_test),
                 verbose=1)

score = model.evaluate(DX_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



for j in range(100):
    predict = model.predict(DX_train)
    #predict = scaler.inverse_transform(predict[j])
    #predictions = Y_train
    predict = predict.astype(int)
    dataframe = pd.DataFrame(predict[j], columns= ['x','y'])
    dataframe.to_csv("./model/test_{}".format(j), index=False)
