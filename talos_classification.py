# tune hyperparameters on multi-class classification task on IvyGAP subset using DenseNet as example
# 
# 2020.04.11 Chenyue Lu

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
#from tensorflow.keras.backend import set_session
# replaced the following line because I got error "numpy() is only available when eager execution is enabled."
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
config = tf.compat.v1.ConfigProto()
set_session(tf.compat.v1.Session(config=config))

hdf5_path = '/home/cl427/results/seg_expts/subsample_0326.hdf5'
subtract_mean = False
exp_id = '37'
loss_fx = 'categorical_crossentropy'
metrics = ['accuracy']
np_seed = 23
tf_seed = 23
model_pathname = '/home/cl427/results/seg_expts/exp_' + exp_id+'/'
print(f"Will save to {model_pathname}")

## import tables and number to process the hdf5 file
import tables
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(1)
# open the hdf5 file we created
hdf5_file = tables.open_file(hdf5_path, mode='r')
#num_classes = len(set(hdf5_file.root.train_labels))
train_labels = np.array(hdf5_file.root.train_labels[:,])
val_labels = np.array(hdf5_file.root.val_labels[:,])
test_labels = np.array(hdf5_file.root.test_labels[:,])

subtract_mean = False
# subtract the training mean
if subtract_mean:
    mm = hdf5_file.root.train_mean[0]
    mm = mm[np.newaxis, ...]


# subset images based on labels
sub_train_indices = np.where(np.in1d(train_labels,[0,1,3,8,9]))[0]
sub_train_images = hdf5_file.root.train_img[sub_train_indices,:,:,:]
sub_train_labels = train_labels[sub_train_indices]
sub_train_labels[sub_train_labels ==3] = 2
sub_train_labels[sub_train_labels ==8] = 3
sub_train_labels[sub_train_labels ==9] = 4
y_train=to_categorical(sub_train_labels).astype('int')

sub_val_indices = np.where(np.in1d(val_labels,[0,1,3,8,9]))[0]
sub_val_images = hdf5_file.root.val_img[sub_val_indices,:,:,:]
sub_val_labels = val_labels[sub_val_indices]
sub_val_labels[sub_val_labels ==3] = 2
sub_val_labels[sub_val_labels ==8] = 3
sub_val_labels[sub_val_labels ==9] = 4
y_val = to_categorical(sub_val_labels).astype('int')

num_classes = len([0,1,3,8,9])

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Masking
from keras.layers import Input
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras import backend as k
from sklearn.preprocessing import normalize
from talos.utils import lr_normalizer
from keras.applications import DenseNet121
from keras.optimizers import RMSprop, Adam, SGD
import talos as ta

# define the range of hyperparameters
p = {
    'batch_size': [32],
    'lr': [0.000001, 0.00001, 0.0001, 0.001,0.01, 0.1],
    'optimizer':[RMSprop, Adam, SGD],
    'class_weight': [{0:1.,1:1.,2:1.,3:1.,4:1.},{0: 1/len(sub_train_labels[sub_train_labels == 0]),1: 1/len(sub_train_labels[sub_train_labels == 1]),2: 1/len(sub_train_labels[sub_train_labels == 2]),3: 1/len(sub_train_labels[sub_train_labels == 3]),4: 1/len(sub_train_labels[sub_train_labels == 4])}]
    }
print(f"parameters tested: {p}")

# compile the model, using mean_squared_error as the loss function
# we could easily swap the loss function to others
from keras.optimizers import *
from keras import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint

## function imageLoader
# load the images from HDF5 file
# will be used in model.fit_generator
def imageLoader(img, labels, batch_size):
    datasetLength = labels.shape[0]
    #datasetLength = len(labels)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < datasetLength:
            limit = min(batch_end, datasetLength)
            X = img[batch_start:limit]
            Y = labels[batch_start:limit]
            yield (X,Y)
            batch_start += batch_size
            batch_end += batch_size

## function talos_model
# define the base model for hyperparameter search
# using InceptionV3 as the base model in this example
def talos_model(sub_train_images, y_train, sub_val_images, y_val, params):
    print(f"parameters: {params}")
    print(f"y_train.shape: {y_train.shape}")
    #input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'
    base_model = DenseNet121(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='max', verbose=1, save_best_only=True)

    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
            loss=loss_fx,
            metrics=metrics,
            class_weight=class_weight)

    out = model.fit_generator(
        imageLoader(sub_train_images,y_train,params['batch_size']),
        steps_per_epoch=sub_train_images.shape[0] // params['batch_size'],
        epochs=20,
        validation_data=imageLoader(sub_val_images,y_val,params['batch_size']),
        validation_steps=sub_val_images.shape[0] // params['batch_size'],
        callbacks=[es,mc],
        verbose=2)

    #print(f"out:{out.history.keys()}")
    return out, model

#print(f"y_train[0]: {y_train[0]}")
# hyperparameter optimization
t = ta.Scan(x=sub_train_images, y=y_train, x_val=sub_val_images, y_val=y_val, params=p, model=talos_model,experiment_name = 'exp_'+exp_id)

ta.Deploy(t,exp_id, metric="val_accuracy", asc=False)

