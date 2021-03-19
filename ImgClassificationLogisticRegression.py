'''
A custom logistic regression model for classifying cat and dog images
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import cv2
import os
from random import shuffle
from PIL import Image
from sklearn.model_selection import train_test_split
from LogisticRegssionImplement import LogisticRegression

train_cat = "catdog/training_set/cats"
test_cat = "catdog/test_set/cats"
train_dog = "catdog/training_set/dogs"
test_dog = "catdog/test_set/dogs"

def prepare_dataset():
    test_data_cat = []
    train_data_cat = []
    test_data_dog = []
    train_data_dog = []
    for img in os.listdir(train_cat):
        path = os.path.join(train_cat, img)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (128,128))
        train_data_cat.append(img1)
    for img in os.listdir(train_dog):
        path = os.path.join(train_dog, img)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (128,128))
        train_data_dog.append(img1)
    for img in os.listdir(test_cat):
        path = os.path.join(test_cat, img)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (128,128))
        test_data_cat.append(img1)
    for img in os.listdir(test_dog):
        path = os.path.join(test_dog, img)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (128,128))
        test_data_dog.append(img1)
    print(len(train_data_cat),len(test_data_cat),len(train_data_dog),len(test_data_dog))
    train_data = np.concatenate((np.asarray(train_data_cat), np.asarray(train_data_dog)),axis=0)
    test_data = np.concatenate((np.asarray(test_data_cat), np.asarray(test_data_dog)),axis=0)
    x_data = np.concatenate((train_data, test_data), axis=0)
    z1 = np.zeros(4000)
    o1 = np.ones(4005)
    z2 = np.zeros(1011)
    o2 = np.zeros(1012)
    y_data = np.concatenate((z1,o1,z2,o2), axis=0).reshape(x_data.shape[0],1)
    return x_data,y_data

x_data, y_data = prepare_dataset()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
number_of_train = x_train.shape[0]
number_of_test = x_test.shape[0]
x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])
x_test_flatten = x_test .reshape(number_of_test,x_test.shape[1]*x_test.shape[2])
x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_test = y_test.T
y_train = y_train.T

lr = LogisticRegression()
lr.logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, no_iteration = 100)

