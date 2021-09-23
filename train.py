import numpy as np
import pandas as pd
import os
import PIL
import PIL.Image
import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import random
from tensorflow.keras import Sequential
from prepare_data import get_images_and_labels
NUM_CLASSES=2
data_root_dir="val/"
from tensorflow.keras import Sequential
from    tensorflow import  keras
from    tensorflow.keras import datasets, layers, optimizers, models
from    tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical





if __name__ == '__main__':
    
    #get the dataset
    training_tensor,all_image_label=get_images_and_labels(data_root_dir)
    
  


    model=Sequential()
    model.add(layers.Conv2D(16, kernel_size=(3, 3),kernel_regularizer=keras.regularizers.l2(0.001),activation='linear',padding='same',input_shape=(180,180,1)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2),padding='same'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(32, (3, 3),kernel_regularizer=keras.regularizers.l2(0.001), activation='linear',padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3),kernel_regularizer=keras.regularizers.l2(0.001), activation='linear',padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))


    model.add(layers.Flatten())
    model.add(layers.Dense(64,kernel_regularizer=keras.regularizers.l2(0.001),activation='linear'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(NUM_CLASSES,kernel_regularizer=keras.regularizers.l2(0.001),activation='softmax'))


    model.summary()
    
    #define learing rate
    lr_schedule=keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,decay_steps=100,decay_rate=0.96,staircase=True
    )

#define optimizer
    my_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=my_optimizer,metrics=['sparse_categorical_accuracy']) 

    all_image_label=np.array(all_image_label)

    model.fit(np.array(training_tensor), all_image_label, 
          batch_size=4,
          epochs=50,
          validation_split=0.2,
          )

    model.summary()