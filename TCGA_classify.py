### transfer best models from before June 2020
### VGG16, InceptionV3, Resnet, Densenet

from datetime import datetime
import tables
import numpy as np
import pandas as pd
import h5py
from os import listdir
import os
from os.path import isfile, join
from keras.models import load_model
import csv
from random import shuffle
import random
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


# set up the GPU session
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.28 # Allowing 28% of GPU memory to be generated to task at hand
set_session(tf.compat.v1.Session(config=config))

#image_dir = '/mnt/data2/datasets/tcgaGBM/tissueImages/1000denseS200/TCGA-02-0004-01A-01-BS1.64d91764-048f-419a-872d-bbd75246fb2e.svs/'
image_dir='/mnt/data2/datasets/tcgaGBM/tissueImages/1000denseS200/'
sample_size = 1000
test_images={}
for dirpath,_,filenames in os.walk(image_dir):
    test_images[dirpath]=[]
    for f in filenames:
        test_images[dirpath].append(os.path.abspath(os.path.join(dirpath, f)))

sample_images=random.sample(test_images.keys(),sample_size)
#test_images = listdir(image_dir)
VGG_model = tf.keras.models.load_model('/home/cl427/results/seg_expts/exp_49/exp_49_VGG_best_model.h5')
Inception_model = tf.keras.models.load_model('/home/cl427/results/seg_expts/exp_49/exp_49_Inception_best_model.h5')
Resnet_model = tf.keras.models.load_model('/home/cl427/results/seg_expts/exp_51/exp_51_ResNet_best_model.h5')
Densenet_model = tf.keras.models.load_model('/home/cl427/results/seg_expts/exp_51/exp_51_DenseNet_best_model.h5')

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
img_dtype = tables.UInt8Atom()

# Theano and Tensorflow organized the data differently
if data_order == 'th':
    data_shape = (0, 3, 224, 224)
elif data_order == 'tf':
    data_shape = (0, 224, 224, 3)
print('data_shape', data_shape)


## ------------ open the specified hdf5 file and create earrays to store the images and train_mean --------------
hdf5_path = '/home/cl427/results/TCGA/TCGA_sample_'+str(sample_size)+'.hdf5'
hdf5_file = tables.open_file(hdf5_path, mode='w')
img_storage = hdf5_file.create_earray(hdf5_file.root, 'imgs', img_dtype, shape=data_shape)

count=0
filenames=[]
for i in range(len(sample_images)):
    image=test_images[sample_images[i]]
    print(sample_images[i])
    for j in range(len(image)):
        count=count+1
        if count % 1000 == 0 and count > 1:
            print('Processed images: {}/{}'.format(count, len(sample_images)*200))
        img = plt.imread(test_images[sample_images[i]][j])
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img_storage.append(img[None])
        filenames.append(test_images[sample_images[i]][j])
    print(count)

hdf5_file.create_array(hdf5_file.root, "filenames", np.array(filenames))

hdf5_file.close()

# read in images from the hdf5 file and predict 


hdf5_file = tables.open_file(hdf5_path, mode='r')
images = hdf5_file.root.imgs[:,:,:,:]
filenames = hdf5_file.root.filenames[:,]

VGG_test_predict = VGG_model.predict(images)
VGG_res = pd.DataFrame(VGG_test_predict)
#VGG_res['VGG_label']=VGG_res.idxmax(axis=1)
VGG_res['filename']=filenames.astype(str)
VGG_res['model']=np.repeat('VGG',len(filenames))

Inception_test_predict = Inception_model.predict(images)
Inception_res = pd.DataFrame(Inception_test_predict)
#Inception_res['Inception_label']=Inception_res.idxmax(axis=1)
Inception_res['filename']=filenames.astype(str)
Inception_res['model']=np.repeat('Inception',len(filenames))

Resnet_test_predict = Resnet_model.predict(images)
Resnet_res = pd.DataFrame(Resnet_test_predict)
#Resnet_res['Resnet_label']=Resnet_res.idxmax(axis=1)
Resnet_res['filename']=filenames.astype(str)
Resnet_res['model']=np.repeat('Resnet',len(filenames))

Densenet_test_predict = Densenet_model.predict(images)
Densenet_res = pd.DataFrame(Densenet_test_predict)
#Densenet_res['Densenet_label']=Densenet_res.idxmax(axis=1)
Densenet_res['filename']=filenames.astype(str)
Densenet_res['model']=np.repeat('Densenet',len(filenames))

#res=pd.concat([VGG_res['VGG_label'],Inception_res['Inception_label'],Resnet_res['Resnet_label'],Densenet_res['Densenet_label']],axis=1)
#res['filename']=filenames

res= pd.concat([VGG_res,Inception_res,Resnet_res,Densenet_res],axis=0,join='outer')
res.to_csv('/home/cl427/results/TCGA/predict_sample_'+str(sample_size)+'_fullpred.csv')

