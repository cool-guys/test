import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed,CuDNNLSTM,BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import keras
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 
import time
from imageloader import data_process
cb_checkpoint = ModelCheckpoint(filepath='model.hdf5',
                                verbose=1)

j = 0
x_data = []
y_data = []
z_data = []
x_img = []
xt_img = []
scaler = MinMaxScaler((0,100))

dp = data_process('./DATA/test')
dp.point_data_load()
dp.image_make()
#dp.image_read()
dp.data_shuffle()



size = int(np.size(dp.point,0) * 0.9)
X_train = dp.point[:size]
X_test = dp.point[size:]
X_train_img = dp.images[:size]
X_test_img = dp.images[size:]
Y_train = dp.label[:size]
Y_test = dp.label[size:]

Y_train = keras.utils.to_categorical(Y_train,num_classes=10, dtype='float32')
Y_test = keras.utils.to_categorical(Y_test,num_classes=10, dtype='float32')

for i in range(np.size(X_train,0)):
  X_train[i] = scaler.fit_transform(X_train[i])
'''
  for j in range(np.size(X_train[i],0)):
    X_train[i][j][0] -= np.mean(X_train[i], axis=0)[0]
    X_train[i][j][1] -= np.mean(X_train[i], axis=0)[1]
    X_train[i][j][0] /= np.std(X_train[i],axis=0)[0]
    X_train[i][j][1] /= np.std(X_train[i],axis=0)[1]
''' 
for i in range(np.size(X_test,0)):
  X_test[i] = scaler.fit_transform(X_test[i])
'''
  for j in range(np.size(X_test[i],0)):
    X_test[i][j][0] -= np.mean(X_test[i], axis=0)[0]
    X_test[i][j][1] -= np.mean(X_test[i], axis=0)[1]
    X_test[i][j][0] /= np.std(X_test[i],axis=0)[0]
    X_test[i][j][1] /= np.std(X_test[i],axis=0)[1]
'''

'''
for i in range(np.size(X_train,0)):
  img = np.zeros((500, 500, 1), np.uint8)
  for k in range(len(X_train[i])):
    if(k != len(X_train[i])-1):
      cv2.line(img, (X_train[i][k][0],X_train[i][k][1]), (X_train[i][k+1][0],X_train[i][k+1][1]), (255,255,255), 25)
  img = cv2.flip(img, 1)
  img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
  x_img.append(img)
X_train_img = np.array(x_img)
X_train_img = np.reshape(X_train_img,(-1,28,28,1))
X_train_img = X_train_img/255
for i in range(np.size(X_test,0)):
  img = np.zeros((500, 500, 1), np.uint8)
  for k in range(len(X_test[i])):
    if(k != len(X_test[i])-1):
      cv2.line(img, (X_test[i][k][0],X_test[i][k][1]), (X_test[i][k+1][0],X_test[i][k+1][1]), (255,255,255), 25)
  img = cv2.flip(img, 1)
  img = cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA)
  xt_img.append(img)
X_test_img = np.array(xt_img)
X_test_img = np.reshape(X_test_img,(-1,28,28,1))
X_test_img = X_test_img/255
'''

'''
for i in range(np.size(X_train,0)):
  X_train[i] = scaler.fit_transform(X_train[i])
  
for i in range(np.size(X_test,0)):
  X_test[i] = scaler.fit_transform(X_test[i])
'''

def train_generator():
  n = 0
  while True:
    x_train = np.reshape(X_train[n],(1,np.size(X_train[n],0),2))


    y_train = Y_train[n].reshape(1,10)
    #print(y_train)
    if(n < np.size(X_train,0)-1):
      n += 1
    else:
      n = 0
    yield x_train, y_train

def test_generator():
  n = 0
  while True:
    x_test = np.reshape(X_test[n],(1,np.size(X_test[n],0),2))

    y_test = Y_test[n].reshape(1,10)
    #print(y_train)
    if(n < np.size(X_test,0)-1):
      n += 1
    else:
      n = 0
    yield x_test, y_test 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Sequential()

model.add(CuDNNLSTM(256, input_shape=(None, 2)))
model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


results = model.fit_generator(train_generator() ,steps_per_epoch=2100, epochs=20,validation_data=test_generator(),validation_steps=900, verbose=1)

score = model.evaluate_generator(test_generator(),steps=900)
scores = model.predict_generator(test_generator(),steps=900)
true_value = np.argmax(Y_test,1)
predict_value = np.argmax(scores,1)
list_ = []
for i in range(np.size(true_value,0)):
  if(true_value[i] != predict_value[i]):
    print('t:{},p:{}'.format(true_value[i],predict_value[i]))
    list_.append(i)

print(score[1]*100)

ROW = 4
COLUMN = 5
j = 1
for i in list_:
    if(j> 20):
      j = 20
    # train[i][0] is i-th image data with size 28x28
    image = X_test_img[i].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, j)         # subplot with size (width 3, height 5)
    j +=1
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    plt.title('predict = {}'.format(np.argmax(scores[i],0)))
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig('mnist_plot.png')
plt.show()
'''
fig = plt.figure(figsize=(10,10))
for i in range(30):
        # 2x5 그리드에 i+1번째 subplot을 추가하고 얻어옴
        subplot = fig.add_subplot(5, 6, i + 1)
 
        # x, y 축의 지점 표시를 안함
        subplot.set_xticks([])
        subplot.set_yticks([])
 
        # subplot의 제목을 i번째 결과에 해당하는 숫자로 설정
        subplot.set_title('%d' % np.argmax(scores[i],0),fontsize=15)
 
        # 입력으로 사용한 i번째 테스트 이미지를 28x28로 재배열하고
        # 이 2차원 배열을 그레이스케일 이미지로 출력
        subplot.imshow(X_test_img[i].reshape((28, 28)),
            cmap=plt.cm.gray_r)
plt.tight_layout()
plt.savefig('lstm_plot.png')
plt.show()
print(results.history.keys())
# summarize history for accuracy
plt.figure(figsize=(10,5))
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('lstm_acc_plot.png')
plt.show()
# summarize history for loss
plt.figure(figsize=(10,5))
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('lstm_loss_plot.png')
plt.show()
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
