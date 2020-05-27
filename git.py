from tkinter import Tk,Label,Button,Entry
from tkinter.filedialog import askopenfilename
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
import cv2
import os
import numpy as np
import pandas as np

import matplotlib.pyplot as plt
class cvn:
    
    
    cnn = Sequential()
    

    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))


    cnn.add(MaxPooling2D(pool_size = (2, 2)))


    cnn.add(Conv2D(32, (3, 3), activation="relu"))

    cnn.add(MaxPooling2D(pool_size = (2, 2)))


    cnn.add(Flatten())

    cnn.add(Dense(activation = 'relu', units = 128))
    cnn.add(Dense(activation = 'sigmoid', units = 1))

    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.

    training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

    validation_generator = test_datagen.flow_from_directory('val',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

    test_set = test_datagen.flow_from_directory('test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
    history = cnn.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 10,
                         validation_data = validation_generator,
                         validation_steps = 62)
    
    save_path="./project/"
    path="./project/"
    cnn.save(os.path.join(save_path,"cnn2.h5"))



    

CATEGORIES = ["HE IS NORMAL", "HE MAY HAVE PNEUMONIA"]
def prepare(filepath):
    IMG_SIZE = 64 
    img_array = cv2.imread(filepath)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)  # return the image with shaping that TF wants.
    
def DisplayDir():
    feedback = askopenfilename()
    Label(root,text=feedback).pack()
    img_array=cv2.imread(feedback , cv2.IMREAD_GRAYSCALE)
    prediction = model.predict([prepare(feedback)])  # Path to PREDICT  
    Label(root,text=CATEGORIES[int(prediction[0][0])]).pack()   
    
root= Tk()
root.title("Pneumonia Diagnosis Assistance System")# title of window
root.geometry("500x500+300+50") # width x Height + position right + position down
obj= cvn()
model = tf.keras.models.load_model("project/cnn2.h5")
label1=Label(root,text='Please upload the X- ray image')
label1.pack()
Button(root, text='Browse', command=DisplayDir).pack()
#label2=Label(root,text='Prediction answer').pack()
root.mainloop()
