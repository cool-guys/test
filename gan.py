import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os
import pandas as pd
#from __future__ import print_function, division
from music21 import converter, instrument, note, chord, stream
import keras
from keras.layers import Input, Dense, Reshape, Dropout, CuDNNLSTM, Bidirectional, LSTM
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class GAN():
    def __init__(self, rows):
        self.seq_length = rows
        self.seq_shape = (self.seq_length, 2)
        self.latent_dim = 50
        self.disc_loss = []
        self.gen_loss =[]
        
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates note sequences
        z = Input(shape=(50,2))
        generated_seq = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(generated_seq)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):

        model = Sequential()
        model.add(LSTM(64, input_shape=(50,2), return_sequences=True))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        seq = Input(shape=(50,2))
        validity = model(seq)

        return Model(seq, validity)
      
    def build_generator(self):

        model = Sequential()
        model.add(LSTM(128, input_shape=(50,2), return_sequences=True))
        model.add(LSTM(128,return_sequences=True))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(2, activation='tanh'))
        model.summary()
        
        noise = Input(shape=(50,2))
        seq = model(noise)

        return Model(noise, seq)

    def train(self,X_train,Y_train, epochs, batch_size=128, sample_interval=50):

        # Load and convert the data

        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        n = 0
        # Training the model
        for epoch in range(epochs):

            
            y_train = Y_train[n].reshape(1,10)
            if(n < np.size(X_train,0)-1):
                n += 1
            else:
                n = 0

            # Training the discriminator
            # Select a random batch of note sequences
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]

            #noise = np.random.choice(range(484), (batch_size, self.latent_dim))
            #noise = (noise-242)/242
            noise = np.random.uniform(0,1,[10,50,2])

            # Generate a batch of new note sequences
            gen_seqs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            #  Training the Generator
            noise = np.random.uniform(0,1,[10,50,2])

            # Train the generator (to have the discriminator label samples as real)
            g_loss = self.combined.train_on_batch(noise, real)

            # Print the progress and save into loss lists
            if epoch % sample_interval == 0:
              print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
              self.disc_loss.append(d_loss[0])
              self.gen_loss.append(g_loss)
        
        self.generate_notes()
        #self.plot_loss()
        
    def generate(self, input_notes):
        # Get pitch names and store in a dictionary
        notes = input_notes
        pitchnames = sorted(set(item for item in notes))
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        
        # Use random noise to generate sequences
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        predictions = self.generator.predict(noise)
        
        pred_notes = [x*242+242 for x in predictions[0]]
        pred_notes = [int_to_note[int(x)] for x in pred_notes]
        
        create_midi(pred_notes, 'gan_final')

    def generate_notes(self):
        for i in range(10):
            noise = np.random.uniform(0,1,[10,50,2])
            predictions = self.generator.predict(noise)
            #predictions = np.reshape(predictions,(10,50,2))
            for j in range(10):
                predict = scaler.inverse_transform(predictions[j])
                dataframe = pd.DataFrame(predict, columns= ['x','y'])
                dataframe.to_csv("./model/test_{}".format(i*10+j), index=False)

        
    def plot_loss(self):
        plt.plot(self.disc_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.close()

if __name__ == '__main__':
    

    j = 0
    x_data = []
    y_data = []

    scaler = MinMaxScaler((0,1))

    for i in range(10):
        while(os.path.exists("./DATA/{}/{}train_{}".format(i,i,j))):
            j += 1
        for k in range(j):
            df = pd.read_csv("./DATA/{}/{}train_{}".format(i,i,k))

            x_data.append(scaler.fit_transform(df[['x','y']].to_numpy()))
            y_data.append(df[['label']].to_numpy())

        
    X_DATA = np.array(x_data)
    Y_DATA = np.array(y_data)

    for i in range(np.size(Y_DATA,0)):
        Y_DATA[i] = np.unique(Y_DATA[i],axis=0)
    Y_DATA = Y_DATA.reshape((np.size(Y_DATA,0),1))


    X_train, X_test, Y_train, Y_test = train_test_split(X_DATA, Y_DATA, test_size=0.1, random_state=42)
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=50, padding='post', dtype='float32')
    print(X_train[0].shape)
    Y_train = keras.utils.to_categorical(Y_train,num_classes=10, dtype='float32')

    Y_test = keras.utils.to_categorical(Y_test,num_classes=10, dtype='float32')


    gan = GAN(rows=50)    
    gan.train(X_train=X_train,Y_train=Y_train,epochs=1, batch_size=10, sample_interval=1)
