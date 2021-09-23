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




##############

def shuffle_dict(original_dict):
    keys = []
    shuffled_dict = {}
    for k in original_dict.keys():
        keys.append(k)
    random.shuffle(keys)
    for item in keys:
        shuffled_dict[item] = original_dict[item]
    return shuffled_dict


def get_images_and_labels(data_root_dir):
    data_root=pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    label_names = sorted(item.name for item in data_root.glob('*/'))
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]                                       

    randnum=random.randint(0,100)
    random.seed(randnum)
    random.shuffle(all_image_path)
    random.seed(randnum)
    random.shuffle(all_image_label)

    miz = PIL.Image.open(all_image_path[0])
    miz=miz.resize((180,180), Image.ANTIALIAS)
    miz = miz.convert("L")
    new = np.asarray(miz)
    new_normalised = new/255
    training_tensor = np.expand_dims(new_normalised, axis=0)

    for names in all_image_path[1:]:
        image_path=names
        miz = PIL.Image.open(image_path)
        miz = miz.convert("L")
        miz=miz.resize((180,180), Image.ANTIALIAS)
        new = np.asarray(miz)
        new_normalised = new/255
        new_tensor = np.expand_dims(new_normalised, axis=0)
        training_tensor = np.concatenate((training_tensor, new_tensor), axis=0)
        print(names)
    training_tensor = np.expand_dims(training_tensor, axis=3)
    print(training_tensor.shape)
    print(all_image_label)
    print('Training shape:', training_tensor.shape)
    print(training_tensor.shape[0], 'sample,',training_tensor.shape[1] ,'x',training_tensor.shape[2] ,'size grayscale image.\n')
    return training_tensor,all_image_label