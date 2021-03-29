'''
AUTOENCODER

-Type of feedforward neural network where the input is same as output
-They compress input to lower dimensional code and reconstruct output from this representation
-To build autoencoder we need 3 components: encoding method, decoding method and loss function to compare output and target
-It is basically a dimensionality reduction/compression algorithm with some properties:
    -They can meaningfully compress data similar to what they have been trained on
    -It is lossy compression
    -They are not supervised, they are self-supervised because they generate their own labels from training data
-The decoder architecture is the mirror image of encoder

Parameters:
-Code size: number of nodes in middle layer. Smaller size results in more compression
-Number of layers: can be as deep
-Number of nodes per layer: stacked encoder, layers stacked over one another, nodes per layer decreases with each subsequent layer
-Loss function: either mse(squared error) ooor binary crossentropy(when in [0,1])

-It can be used in denoising, where the input image will be a noised one and target will be denoised one
-It is also used in dimensionality reduction
'''

from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.regularizers import l1
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class AutoEncoder:

    def __init__(self):
        self.input_size = 784
        self.hidden_size = 128
        self.code_size = 32
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def processData(self):
        self.x_train = self.x_train.astype('float32')/255.0
        self.x_test = self.x_test.astype('float32')/255.0
        self.x_train = self.x_train.reshape((len(self.x_train), np.prod(self.x_train.shape[1:])))
        self.x_test = self.x_test.reshape((len(self.x_test), np.prod(self.x_test.shape[1:])))
        noise_factor = 0.4
        x_train_noisy = self.x_train + noise_factor*np.random.normal(size=self.x_train.shape)
        x_test_noisy = self.x_test + noise_factor*np.random.normal(size=self.x_test.shape)
        #to restrict values in [0,1]
        self.x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
        self.x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

    def defineModel(self):
        input_img = Input(shape=(self.input_size,))
        hidden_1 = Dense(self.hidden_size, activation='relu')(input_img)
        code = Dense(self.code_size, activation='relu')(hidden_1)
        hidden_2 = Dense(self.hidden_size, activation='relu')(code)
        output_img = Dense(self.input_size, activation='sigmoid')(hidden_2)
        self.autoencoder = Model(input_img, output_img)

    def trainModel(self):
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        self.autoencoder.fit(self.x_train_noisy, self.x_train, epochs=3)

    def plotResults(self):
        n = 3
        images = self.autoencoder.predict(self.x_test_noisy)
        for i in range(n):
            ax = plt.subplot(3,n,i+1)
            plt.imshow(self.x_test[i].reshape(28,28))
            ax = plt.subplot(3,n,i+1+n)
            plt.imshow(self.x_test_noisy[i].reshape(28,28))
            ax = plt.subplot(3,n,i+1+2*n)
            plt.imshow(images[i].reshape(28,28))
            plt.show()

autoEncoder = AutoEncoder()
autoEncoder.processData()
autoEncoder.defineModel()
autoEncoder.trainModel()
autoEncoder.plotResults()