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
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D,Conv1D
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
scaler = MinMaxScaler((0,1))

for i in range(10):
  while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
    j += 1
  for k in range(j):
    df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,k))
    df_img = df[['x','y']].to_numpy()
    img = np.zeros((500, 500, 1), np.uint8)
    for k in range(len(df_img)):
      if(k != len(df_img)-1):
        cv2.line(img, (df_img[k][0],df_img[k][1]), (df_img[k+1][0],df_img[k+1][1]), (255,255,255), 30)
    img = cv2.flip(img, 1)
    img = cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA)

    x_data.append(img)
    y_data.append(scaler.fit_transform(df[['x','y']].to_numpy()))

    
X_DATA = np.array(x_data)
X_DATA = np.reshape(X_DATA,(-1,64,64,1))
X_DATA = X_DATA/255
Y_DATA = np.array(y_data)






X_train, X_test, Y_train, Y_test = train_test_split(X_DATA, Y_DATA, test_size=0.1, random_state=16)


X_val = X_train[0:18]
X_train = X_train[18:-1]
DX_val = np.reshape(X_val,(-1,64*64))
DX_train = np.reshape(X_train,(-1,64*64))
DX_test = np.reshape(X_test,(-1,64*64))

Y_val = keras.preprocessing.sequence.pad_sequences(Y_train[0:18], maxlen=45, padding='post', dtype='float32',value=0.5)
Y_train = keras.preprocessing.sequence.pad_sequences(Y_train[18:-1], maxlen=45, padding='post', dtype='float32',value=0.5)
Y_test = keras.preprocessing.sequence.pad_sequences(Y_test, maxlen=45, padding='post', dtype='float32',value=0.5)


#print(Y_train)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def custom_activation(x):
  sig = tf.math.sigmoid(x) * 64
  tf.cast(sig,tf.int32)
  return sig

input_1 = Input(shape=(64, 64, 1))

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
x_1 = Dense(units=135, activation='relu')(x_1)
x_1 = Dropout(0.5)(x_1)
x_1 = Reshape((45,-1))(x_1)
x_1 = CuDNNLSTM(256, input_shape=(None,2),return_sequences=True)(x_1)

input_2 = Input(shape=(64*64,))

x_2 = Dense(512, activation='relu')(input_2)
x_2 = Dense(512,activation='relu')(x_2)
x_2 = Dense(450, activation='relu')(x_2)
x_2 = Dense(450, activation='relu')(x_2)
x_2 = Reshape((45,-1))(x_2)
x_2 = CuDNNLSTM(256, input_shape=(None,2),return_sequences=True)(x_2)

merged = concatenate([x_1,x_2])
m = Dense(512, activation='relu')(merged)
#m = Reshape((45,-1))(m)
#m = CuDNNLSTM(256, input_shape=(None,2),return_sequences=True)(m)
#m = Dense(256, activation='relu')(m)
m = Dense(2, activation='sigmoid')(m)

'''
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization(momentum=0.4))
model.add(Conv2D(64, (1, 1)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(momentum=0.4))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization(momentum=0.4))
model.add(Conv2D(64, (1, 1)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))



model.add(Flatten())
model.add(Dense(400))
model.add(Dropout(0.25))
model.add(Reshape((20,-1)))
model.add(CuDNNLSTM(128, input_shape=(25,2), return_sequences=True))
model.add(BatchNormalization())
model.add(Dense(2, activation=custom_activation))
'''
'''
model = Sequential()
model.add(Conv1D(filters=128,kernel_size=8,strides=1,activation='relu',padding='same',input_shape=(45,45)))
model.add(Conv1D(filters=128,kernel_size=8,strides=1,activation='relu',padding='same'))

model.add(Conv1D(filters=64,kernel_size=8,strides=1,activation='relu',padding='same'))
model.add(Conv1D(filters=64,kernel_size=8,strides=1,activation='relu',padding='same'))

model.add(Conv1D(filters=32,kernel_size=8,strides=1,activation='relu',padding='same'))
model.add(Conv1D(filters=32,kernel_size=8,strides=1,activation='relu',padding='same'))

model.add(CuDNNLSTM(256, input_shape=(45,2), return_sequences=True))
model.add(Dense(128))
model.add(Dense(2, activation='sigmoid'))
'''
model = Model(inputs=[input_1, input_2], outputs = m)
model.summary()

early_stopping = EarlyStopping(patience = 200)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit([X_train,DX_train], Y_train,
                 batch_size=16,
                 epochs=1000,
                 validation_data=([X_val,DX_val], Y_val),
                 verbose=1,
                 callbacks=[cb_checkpoint])

score = model.evaluate([X_test,DX_test], Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



for j in range(40):
    predict = model.predict([X_train,DX_train])
    scaler.inverse_transform(predict[j])
    #predictions = Y_train
    predict = predict.astype(int)
    dataframe = pd.DataFrame(predict, columns= ['x','y'])
    dataframe.to_csv("./model/test_{}".format(j), index=False)
