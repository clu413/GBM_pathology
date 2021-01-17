# uses 5-class classification on 0.1% subset of IvyGap tiles 
# run on all four architectures in one script to minimize changes and inconsistencies.
# also saves time from loading up images from hdf5

import pandas as pd
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from datetime import datetime
import tables
import numpy as np
import random
import matplotlib.pyplot as plt
import h5py
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, LSTM, Activation, Masking, Input
from keras import backend as k
from sklearn.preprocessing import normalize
from keras import losses
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications import DenseNet121
from keras.optimizers import RMSprop, Adam, SGD

# set up the GPU session
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.28 # Allowing 28% of GPU memory to be generated to task at hand
set_session(tf.compat.v1.Session(config=config))

hdf5_path = '/home/cl427/results/seg_expts/subsample_0326.hdf5'
subtract_mean = False
# set the hyperparameters
batch_size = 64
exp_id = '49'
loss_fx = 'categorical_crossentropy'
metrics = ['accuracy']
np_seed = 23
tf_seed = 23
model_pathname = '/home/cl427/results/seg_expts/exp_' + exp_id
print(f"Will save to {model_pathname}")
print(f"Start time is: {datetime.now()}")

random.seed(23)
# open the hdf5 file we created
hdf5_file = tables.open_file(hdf5_path, mode='r')

sub_train_indices = np.where(np.in1d(np.array(hdf5_file.root.train_labels[:,]),[0,1,3,8,9]))[0]
sub_train_images = hdf5_file.root.train_img[sub_train_indices,:,:,:]
sub_train_labels = np.array(hdf5_file.root.train_labels[:,])[sub_train_indices]
sub_train_labels[sub_train_labels ==3] = 2
sub_train_labels[sub_train_labels ==8] = 3
sub_train_labels[sub_train_labels ==9] = 4
print('sub_train_labels',sub_train_labels)

sub_val_indices = np.where(np.in1d(np.array(hdf5_file.root.val_labels[:,]),[0,1,3,8,9]))[0]
sub_val_images = hdf5_file.root.val_img[sub_val_indices,:,:,:]
sub_val_labels = np.array(hdf5_file.root.val_labels[:,])[sub_val_indices]
sub_val_labels[sub_val_labels ==3] = 2
sub_val_labels[sub_val_labels ==8] = 3
sub_val_labels[sub_val_labels ==9] = 4
print('sub_val_labels',sub_val_labels)

sub_test_indices = np.where(np.in1d(np.array(hdf5_file.root.test_labels[:,]),[0,1,3,8,9]))[0]
sub_test_images = hdf5_file.root.test_img[sub_test_indices,:,:,:]
sub_test_labels = np.array(hdf5_file.root.test_labels[:,])[sub_test_indices]
sub_test_labels[sub_test_labels ==3] = 2
sub_test_labels[sub_test_labels ==8] = 3
sub_test_labels[sub_test_labels ==9] = 4
print('sub_test_labels',sub_test_labels)

class_weight = {0: 1/len(sub_train_labels[sub_train_labels == 0]),1: 1/len(sub_train_labels[sub_train_labels == 1]),2: 1/len(sub_train_labels[sub_train_labels == 2]),3: 1/len(sub_train_labels[sub_train_labels == 3]),4: 1/len(sub_train_labels[sub_train_labels == 4])}
num_classes = len([0,1,3,8,9])

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

# VGG16 
# talos runs indicated 0.001 is a good learning rate
# divide by 1000 to get actual learning rate
# source: https://github.com/autonomio/talos/blob/master/talos/model/normalizers.py
architecture = 'VGG' # change this!!!!!!
print('architecture',architecture)
random.seed(23)
lr = 0.000001
optimizer = RMSprop

input_tensor = Input(shape=(224, 224, 3))
base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False) # change this chunk!!!!!
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min', baseline=None, restore_best_weights=True)
mc = ModelCheckpoint(model_pathname+'/exp_'+exp_id+'_'+architecture+'_best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=optimizer(lr=lr),loss=loss_fx,metrics=metrics)

History = model.fit_generator(
        imageLoader(sub_train_images, to_categorical(sub_train_labels),batch_size),
        steps_per_epoch=sub_train_images.shape[0] // batch_size,
        epochs=30,
        workers=0,
        class_weight=class_weight,
        validation_data=imageLoader(sub_val_images,to_categorical(sub_val_labels),batch_size),
        validation_steps=sub_val_images.shape[0] // batch_size,
        callbacks=[es,mc],
        verbose = 1)

# save history
hist_df = pd.DataFrame(History.history)
hist_csv_file = model_pathname+'/exp_'+exp_id+'_'+architecture+'_history.csv'
with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


# Plot the loss function
fig, ax = plt.subplots(2, 1, figsize=(10,6))

ax[0].plot(History.history['loss'], 'r', label='train')
ax[0].plot(History.history['val_loss'], 'b' ,label='val')
ax[0].set_xlabel(r'Epoch', fontsize=20)
ax[0].set_ylabel(r'Loss', fontsize=20)
ax[0].legend()
ax[0].tick_params(labelsize=20)

# Plot the accuracy
ax[1].plot(History.history['accuracy'], 'r', label='train')
ax[1].plot(History.history['val_accuracy'], 'b' ,label='val')
ax[1].set_xlabel(r'Epoch', fontsize=20)
ax[1].set_ylabel(r'Accuracy', fontsize=20)
ax[1].legend()
ax[1].tick_params(labelsize=20)

fig.tight_layout()
fig.savefig(model_pathname +'/exp_'+exp_id+'_'+architecture+'_plot.png')
print(f"End time is: {datetime.now()}")

VGG_test_predict = model.predict(sub_test_images)  # change this!!!!!

# InceptionV3
# talos runs indicated 0.01 both 0.001 are good learning rates
# both Adam and RMSprop are good with either learning rate 
# divide by 1000 to get actual learning rate
architecture = 'Inception'
print('architecture: ',architecture)
random.seed(23)
lr = 0.00001
optimizer = RMSprop

base_model = InceptionV3(weights='imagenet', include_top=False) # change this chunk!!!!!
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min', baseline=None, restore_best_weights=True)
mc = ModelCheckpoint(model_pathname+'/exp_'+exp_id+'_'+architecture+'_best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=optimizer(lr=lr),loss=loss_fx,metrics=metrics)

History = model.fit_generator(
        imageLoader(sub_train_images, to_categorical(sub_train_labels),batch_size),
        steps_per_epoch=sub_train_images.shape[0] // batch_size,
        epochs=30,
        workers=0,
        class_weight=class_weight,
        validation_data=imageLoader(sub_val_images,to_categorical(sub_val_labels),batch_size),
        validation_steps=sub_val_images.shape[0] // batch_size,
        callbacks=[es,mc],
        verbose=1)

# save history
hist_df = pd.DataFrame(History.history)
hist_csv_file = model_pathname+'/exp_'+exp_id+'_'+architecture+'_history.csv'
with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

# Plot the loss function
fig, ax = plt.subplots(2, 1, figsize=(10,6))

ax[0].plot(History.history['loss'], 'r', label='train')
ax[0].plot(History.history['val_loss'], 'b' ,label='val')
ax[0].set_xlabel(r'Epoch', fontsize=20)
ax[0].set_ylabel(r'Loss', fontsize=20)
ax[0].legend()
ax[0].tick_params(labelsize=20)

# Plot the accuracy
ax[1].plot(History.history['accuracy'], 'r', label='train')
ax[1].plot(History.history['val_accuracy'], 'b' ,label='val')
ax[1].set_xlabel(r'Epoch', fontsize=20)
ax[1].set_ylabel(r'Accuracy', fontsize=20)
ax[1].legend()
ax[1].tick_params(labelsize=20)

fig.tight_layout()
fig.savefig(model_pathname +'/exp_'+exp_id+'_'+architecture+'_plot.png')

print(f"End time is: {datetime.now()}")

Inception_test_predict = model.predict(sub_test_images)


# ResNet
# talos runs indicated 0.01 as a good learning rate
# divide by 100 to get actual learning rate
architecture = 'ResNet' # change this!!!!!!
print('architecture',architecture)
random.seed(23)
lr = 0.0001
optimizer = SGD

base_model = ResNet50(weights='imagenet', include_top=False) # change this chunk!!!!!
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min', baseline=None, restore_best_weights=True)
mc = ModelCheckpoint(model_pathname+'/exp_'+exp_id+'_'+architecture+'_best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=optimizer(lr=lr),loss=loss_fx,metrics=metrics)

History = model.fit_generator(
        imageLoader(sub_train_images, to_categorical(sub_train_labels),batch_size),
        steps_per_epoch=sub_train_images.shape[0] // batch_size,
        epochs=30,
        workers=0,
        class_weight=class_weight,
        validation_data=imageLoader(sub_val_images,to_categorical(sub_val_labels),batch_size),
        validation_steps=sub_val_images.shape[0] // batch_size,
        callbacks=[es,mc],
        verbose=1)


# save history
hist_df = pd.DataFrame(History.history)
hist_csv_file = model_pathname+'/exp_'+exp_id+'_'+architecture+'_history.csv'
with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

# Plot the loss function
fig, ax = plt.subplots(2, 1, figsize=(10,6))

ax[0].plot(History.history['loss'], 'r', label='train')
ax[0].plot(History.history['val_loss'], 'b' ,label='val')
ax[0].set_xlabel(r'Epoch', fontsize=20)
ax[0].set_ylabel(r'Loss', fontsize=20)
ax[0].legend()
ax[0].tick_params(labelsize=20)

# Plot the accuracy
ax[1].plot(History.history['accuracy'], 'r', label='train')
ax[1].plot(History.history['val_accuracy'], 'b' ,label='val')
ax[1].set_xlabel(r'Epoch', fontsize=20)
ax[1].set_ylabel(r'Accuracy', fontsize=20)
ax[1].legend()
ax[1].tick_params(labelsize=20)

fig.tight_layout()
fig.savefig(model_pathname +'/exp_'+exp_id+'_'+architecture+'_plot.png')
print(f"End time is: {datetime.now()}")

ResNet_test_predict = model.predict(sub_test_images)  # change this!!!!!


# DenseNet
architecture = 'DenseNet' # change this!!!!!!
print('architecture',architecture)
random.seed(23)
lr = 0.000001
optimizer = Adam

base_model = DenseNet121(weights='imagenet', include_top=False) # change this chunk!!!!!
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min', baseline=None, restore_best_weights=True)
mc = ModelCheckpoint(model_pathname+'/exp_'+exp_id+'_'+architecture+'_best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=optimizer(lr=lr),loss=loss_fx,metrics=metrics)

History = model.fit_generator(
        imageLoader(sub_train_images, to_categorical(sub_train_labels),batch_size),
        steps_per_epoch=sub_train_images.shape[0] // batch_size,
        epochs=30,
        workers=0,
        class_weight=class_weight,
        validation_data=imageLoader(sub_val_images,to_categorical(sub_val_labels),batch_size),
        validation_steps=sub_val_images.shape[0] // batch_size,
        callbacks=[es,mc],
        verbose=1)

# save history
hist_df = pd.DataFrame(History.history)
hist_csv_file = model_pathname+'/exp_'+exp_id+'_'+architecture+'_history.csv'
with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


# Plot the loss function
fig, ax = plt.subplots(2, 1, figsize=(10,6))

ax[0].plot(History.history['loss'], 'r', label='train')
ax[0].plot(History.history['val_loss'], 'b' ,label='val')
ax[0].set_xlabel(r'Epoch', fontsize=20)
ax[0].set_ylabel(r'Loss', fontsize=20)
ax[0].legend()
ax[0].tick_params(labelsize=20)

# Plot the accuracy
ax[1].plot(History.history['accuracy'], 'r', label='train')
ax[1].plot(History.history['val_accuracy'], 'b' ,label='val')
ax[1].set_xlabel(r'Epoch', fontsize=20)
ax[1].set_ylabel(r'Accuracy', fontsize=20)
ax[1].legend()
ax[1].tick_params(labelsize=20)

fig.tight_layout()
fig.savefig(model_pathname +'/exp_'+exp_id+'_'+architecture+'_plot.png')
print(f"End time is: {datetime.now()}")

DenseNet_test_predict = model.predict(sub_test_images)  # change this!!!!!

# test on holdout test set and save predictions 
res = pd.DataFrame(VGG_test_predict,Inception_test_predict,ResNet_test_predict,DenseNet_test_predict,sub_test_labels)
#res.index = sub_test_indices # its important for comparison
res.columns = ['VGG','Inception','ResNet','DenseNet','True labels']
res.to_csv(model_pathname+ '/exp_' + exp_id + "_prediction_results.csv")

