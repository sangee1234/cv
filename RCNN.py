import os
import cv2
import keras
import pandas as pd
import numpy as np
from keras.layers import Dense
from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping


class RCNN():

    def __init__(self):
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.train_images = []
        self.train_labels = []
        self.testdata = []
        self.traindata = []
        self.model_final = 0
        self.vggmodel = VGG16(weights='imagenet', include_top=True)

    def get_iou(self, bb1, bb2):

        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        iou = intersection_area / float(bb1_area+bb2_area-intersection_area)
        return iou
    
    def create_training_set(self ):
        for e,i in enumerate(os.listdir("Annotations")):
            if i.startswith("airplane"):
                filename = i.split(".")[0]+".jpg"
                image = cv2.imread(os.path.join("Images",filename))
                imout = image.copy()
                df = pd.read_csv(os.path.join("Annotations",i))
                gtvalues=[]
                for row in df.iterrows():
                    x1 = int(row[1][0].split(" ")[0])
                    y1 = int(row[1][0].split(" ")[1])
                    x2 = int(row[1][0].split(" ")[2])
                    y2 = int(row[1][0].split(" ")[3])
                    gtvalues.append({"x1":x1,"x2":x2,"y1":y1,"y2":y2})
                self.ss.setBaseImage(image)
                self.ss.switchToSelectiveSearchFast()
                ssresults = self.ss.process()
                counter = 0
                falsecounter = 0
                flag = 0
                fflag = 0
                bflag = 0
                for e,result in enumerate(ssresults):
                    #for each returned by selective search(only top 2000)
                    if e<2000 and flag==0:
                        for gtval in gtvalues:
                            #for each annotation and each result 
                            x,y,w,h = result
                            iou = self.get_iou(gtval,{"x1":x,"x2":x+w,"y1":y,"y2":y+h})
                            #collect max 30 pos and neg sample from an image
                            if counter < 30:
                                if iou > 0.70:
                                    timage = imout[y:y+h,x:x+w]
                                    resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                    self.train_images.append(resized)
                                    self.train_labels.append(1)
                                    counter += 1
                            else :
                                fflag =1
                            if falsecounter <30:
                                if iou < 0.3:
                                    timage = imout[y:y+h,x:x+w]
                                    resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                    self.train_images.append(resized)
                                    self.train_labels.append(0)
                                    falsecounter += 1
                            else :
                                bflag = 1
                        if fflag == 1 and bflag == 1:
                            flag = 1

    def model(self):
        #freeze 15 layers of vgg
        for layers in (self.vggmodel.layers)[:15]:
            layers.trainable = False

        X = self.vggmodel.layers[-2].output
        #as output only 2 classes
        predictions = Dense(2, activation="softmax")(X)
        self.model_final = Model(input = self.vggmodel.input,output=predictions)
        opt = Adam(lr=0.0001)
        self.model_final.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
        self.model_final.summary()

    def prep_training_data(self):
        X_new = np.array(self.train_images)
        Y_new = np.array(self.train_labels)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_new, Y_new, test_size=0.1)

        trdata = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rotation_rang = 90)
        self.traindata = trdata.flow(x=self.X_train, y=self.y_train)
        tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
        self.testdata = tsdata.flow(x=self.X_test, y=self.y_test)

    def train(self):
        checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
        hist = model_final.fit_generator(generator= self.traindata, steps_per_epoch= 10, epochs= 1000, validation_data= self.testdata, validation_steps=2, callbacks=[checkpoint,early])

    def test(self):
        im = X_test[100]
        plt.imshow(im)
        img = np.expand_dims(im, axis=0)
        out= self.model_final.predict(img)
        if out[0][0] > out[0][1]:
            print("plane")
        else:
            print("not plane")

    def detect(self):
        for e,i in enumerate(os.listdir("Images")):
            img = cv2.imread(os.path.join(path, i))
            self.ss.setBaseImage(img)
            self.ss.switchToSelectiveSearchFast()
            ssresults = ss.process()
            imout = img.copy()
            for e, result in enumerate(Ssresults):
                if e<2000:
                    x,y,w,h = result
                    im1 = imout[y:y+h,x:x+w]
                    resized = cv2.resize(im1, (224, 224), interpolation= cv2.INTER_AREA)
                    im2 = np.expand_dims(resized, axis = 0)
                    out = self.model_final.predict(im2)
                    if(out[0][0] > 0.65):
                        cv2.rectangle(imout, (x,y), (x+2, y+h), (0,255,0), 1, cv2.LINE_AA)
            plt.figure()
            plt.imshow()
        


rcnn = RCNN()
print("Step 1 done")
rcnn.create_training_set()
print("Step 2 done")
rcnn.model()
print("Step 3 done")
rcnn.prep_training_data()
print("Step 4 done")
rcc.train()
print("Step 5 done")
rcnn.test()
print("Step 6 done")
rcnn.detect()