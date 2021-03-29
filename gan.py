import numpy as np
from matplotlib.pyplot import plt
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
from keras.layers import Reshape

class GAN:

    def __init__(self):
        (self.train_x, self.train_y), (self.test_x,self.test_y) = load_data()

    def discriminatorModel(self):
        #take input image and output as real/fake
        model = Sequential()
        model.add((Conv2D(64,(3,3),strides=(2,2), padding='same',input_shape(28,28,1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64,(3,3),strides=(2,2),padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1,activation='sigmoid'))
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer = opt, metrics=['accuracy'])
        return model

    def generatorModel(self):
        model = Sequential()
        n_nodes = 128*7*7
        model.add(Dense(n_nodes, input_dim=100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape(7,7,128))
        #conv2DTranspose is combinig upsampling and conv2d
        model.add(Conv2DTranspose(128,(4,4),strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(1,(7,7),activation='sigmoid',padding='same'))
        return model

    def ganModel(g_model, d_model):
        d_model.trainable = False
        model = Sequential()
        model.add(g_model)
        moodel.add(d_model)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy',optimizer=opt)
        return model

    #model will be updated in batchesm 1 of real and 1 of generated, 1 epoch is for entire dataset
    def loadRealSamples(self):
        (trainX, _), (_,_) = load_data()
        X= expand_dims(trainX,axis=-1)
        X= X.astype('float32')
        X= X/255.0
        return X

    def generateLatentPoints(self,latent_dim, n_sample):
        x_input = randn(latent_dim*n_sample)
        x_input = x_input.reshape(n_sample, latent_dim)
        return x_input

    def generateRealSamples(self,dataset, n_samples):
	    ix = randint(0, dataset.shape[0], n_samples)
        X = dataset[ix]
        y = ones((n_samples, 1))
	    return X, y

    def generateFakeSamples(self, n_samples):
        X = rand(28 * 28 * n_samples)
        X = X.reshape((n_samples, 28, 28, 1))
        y = zeros((n_samples, 1))
        return X, y

    def generateFakeSamples(self,g_model, latent_dim, n_samples):
        X_input = self.generateLatentPoints(latent_dim, n_samples)
        X = g_model.predict(X_input)
        y = zeros((n_samples, 1))
        return X, y

    def trainDiscriminator(model, dataset, n_iter=100, n_batch=256):
        half_batch = int(n_batch/2)
        for in range(n_iter):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            _, real_acc = model.train_on_batch(X_real,y_real)
            X_fake, y_fake = generate_fake_samples(half_batch)
            _, fake_acc = model.train_on_batch(X_fake, y_fake)

    def discriminatorModelTrain(self):
        model = self.discriminatorModel()
        dataset = self.loadRealSamples()
        self.trainDiscriminator(model, dataset)

    def generateSample(self):
        model = self.generatorModel()
        X, _ = self.generateFakeSamples(model,100,25)

    def createGanModel(self):
        d_model = self.discriminatorModel()
        g_model = self.generatorModel(100)
        gan_model = define_gan(g_model, d_model)

