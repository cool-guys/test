
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

cb_checkpoint = ModelCheckpoint(filepath='model.hdf5',
                                verbose=1)

j = 0
x_data = []
y_data = []
z_data = []
x_img = []
xt_img = []
scaler = StandardScaler()

for i in range(10):
  while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
    j += 1
  for k in range(j):
    df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,k))

    df_img_x = df[['x']].to_numpy()
    df_img_x_mean = np.mean(df_img_x)
    df_img_x = (df_img_x - df_img_x_mean)/np.std(df_img_x)

    df_img_y = df[['y']].to_numpy()
    df_img_y_mean = np.mean(df_img_y)
    df_img_y = (df_img_y - df_img_y_mean)/np.std(df_img_y)

    df_img = np.concatenate((df_img_x,df_img_y), axis=1)

    z_data.append(df_img)
    x_data.append(df[['x','y']].to_numpy())
    y_data.append(df[['label']].to_numpy())

    
X_DATA = np.array(x_data)
Y_DATA = np.array(y_data)
Z_DATA = np.array(z_data)

X_DATA = Z_DATA

for i in range(np.size(Y_DATA,0)):
    Y_DATA[i] = np.unique(Y_DATA[i],axis=0)
Y_DATA = Y_DATA.reshape((np.size(Y_DATA,0),1))


X_train, X_test, Y_train, Y_test = train_test_split(X_DATA, Y_DATA, test_size=0.3, random_state=42)

Y_train = keras.utils.to_categorical(Y_train,num_classes=10, dtype='float32')

Y_test = keras.utils.to_categorical(Y_test,num_classes=10, dtype='float32')
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

model.add(CuDNNLSTM(128, return_sequences=True, input_shape=(None, 2)))
model.add(CuDNNLSTM(32))
model.add(Dense(128))

model.add(LeakyReLU(alpha=0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


results = model.fit_generator(train_generator() ,steps_per_epoch=200, epochs=100,validation_data=test_generator(),validation_steps=60, verbose=1,callbacks=[cb_checkpoint])

score = model.evaluate_generator(test_generator(),steps=60)
scores = model.predict_generator(test_generator(),steps=60)
true = np.argmax(Y_test,1)
pred = np.argmax(scores,1)
print(score[1]*100)
for i in range(60):
  if(true[i] != pred[i]):
    print('t',true[i])
    print('p',pred[i])

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