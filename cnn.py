import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import *
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential,Model
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import matplotlib.pyplot as plt
from imageloader import data_process
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
number = False

columns = ['x','y', 'label']

lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_acc',factor=0.1, patience=20,min_lr=0.0001)
MODEL_SAVE_FOLDER_PATH = './model/cnn_alpha'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + '.hdf5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc',
                                verbose=1, save_best_only=True)
                                
j = 0
x_data = []
y_data = []


def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
 
    return x   
 
    
 
def conv2_layer(x):         
    x = MaxPooling2D((3, 3), 2)(x)     
 
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x
 
 
 
def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x
 
 
 
def conv4_layer(x):
    shortcut = x        
  
    for i in range(6):     
        if(i == 0):            
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return x
 
 
 
def conv5_layer(x):
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return x


#숫자인지 아닌지
number = False


if(number):
  dp = data_process('./DATA/aug/all/train/number',number)
  dp.point_data_load()
  #dp.image_read()
  dp.sequence_64()
  dp.image_make()
  dp.data_shuffle()

  class_num = 10

else:
  dp_train = data_process('./DATA/aug/all/train/Alphabet',number)
  dp_train.point_data_load()
  #dp.image_read()
  dp_train.sequence_64()
  dp_train.image_make()
  dp_train.data_shuffle()
  
  dp_test = data_process('./DATA/aug/all/test/Alphabet',number)
  dp_test.point_data_load()
  #dp.image_read()
  dp_test.sequence_64()
  dp_test.image_make()
  dp_test.data_shuffle()
 
  class_num = 26


X_train = dp_train.images[:]
X_test = dp_test.images[:]
Y_train = dp_train.label[:]
Y_test = dp_test.label[:]


Y_train = keras.utils.to_categorical(Y_train,num_classes=class_num, dtype='float32')
Y_test = keras.utils.to_categorical(Y_test,num_classes=class_num, dtype='float32')

# cudnn 오류 해결용(RTX에서만 생기는 문제로 보임))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding="valid",input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.45))

model.add(Conv2D(64, kernel_size=3, padding="valid"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Dropout(0.45))

model.add(Conv2D(128, kernel_size=3, padding="valid"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Dropout(0.45))

model.add(Conv2D(256, kernel_size=3, padding="valid"))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Dropout(0.45))

model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
#model.add(GlobalAveragePooling2D())
model.add(Dropout(0.45))

model.add(Dense(26,activation='softmax'))


'''
input_1 = Input(shape=(28, 28, 1))

x = conv1_layer(input_1)
x = conv2_layer(x)
#x = conv3_layer(x)
#x = conv4_layer(x)
x = GlobalAveragePooling2D()(x)

x = Dense(26, activation='softmax')(x)

model = Model(input_1, x)
'''
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=0.001,epsilon=1e-04), metrics=['accuracy'])
hist = model.fit(X_train, Y_train,
                batch_size=64,
                validation_data=(X_test,Y_test),
                epochs=250,
                verbose=1,
                callbacks=[cb_checkpoint,lr_reduce])

model = load_model('./model/cnn_alpha.hdf5')

score = model.evaluate(X_test, Y_test, verbose=0)
pred = model.predict(X_test)
true_value = np.argmax(Y_test,1)
predict_value = np.argmax(pred,1)
print(np.argmax(pred,axis=1))
print(score[1]*100)
#print("복원된 모델의 정확도: {:5.2f}%".format(100*score))
list_ = []
for i in range(np.size(true_value,0)):
  if(true_value[i] != predict_value[i]):
    if(number):
      print('t:{},p:{}'.format(true_value[i],predict_value[i]))
    else:
      print('t:{},p:{}'.format(chr(97+true_value[i]),chr(97+predict_value[i])))
    list_.append(i)



ROW = 5
COLUMN = 4
j = 1
for i in list_:
    if(j> 20):
      j = 20
    # train[i][0] is i-th image data with size 28x28
    image = X_test[i].reshape(28, 28)   # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, j)         # subplot with size (width 3, height 5)
    j +=1
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    if(number):
      plt.title('predict = {},{}'.format((np.argmax(pred[i],0)),chr(97 + true_value[i])))
    else:
      plt.title('predict = {},{}'.format(chr(97+ np.argmax(pred[i],0)),chr(97 + true_value[i])))  
    plt.axis('off')  # do not show axis value
plt.tight_layout()   # automatic padding between subplots
plt.savefig('cnn_{}.png'.format(score[1]*100))
plt.show()


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cnn_acc.png')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('cnn_loss.png')
plt.show()

cm = confusion_matrix(true_value, predict_value)
df_cm = pd.DataFrame(cm, index = [i for i in "abcdefghijklmnopqrstuvwxyz"],
                  columns = [i for i in "abcdefghijklmnopqrstuvwxyz"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_map_cnn.png')
plt.show()