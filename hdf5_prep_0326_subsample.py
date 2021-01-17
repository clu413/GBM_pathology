# define the path to hdf5 files, train.txt, test.txt, and image files
hdf5_path = '/home/cl427/GBM/results/train_vs_test_by_pt/subsample_0326.hdf5'

trainInputFilename="/home/cl427/GBM/results/train_vs_test_by_pt/train/train.txt"
valInputFilename="/home/cl427/GBM/results/train_vs_test_by_pt/val/val.txt"
testInputFilename="/home/cl427/GBM/results/train_vs_test_by_pt/test/test.txt"

subsample_percentage = 0.001

# import csv and shuffle
import csv
import random
import math
## function readInputFileName
# read train.txt and test.txt
# format of the files:
# <image names> <targeted value>
#
def readInputFileName(inputFilename, shuffle_data):
    imageFilenames=[]
    imageClasses=[]
    with open(inputFilename,'r') as f:
        reader=csv.reader(f,delimiter=',')
        next(reader, None)
        rows = [row for row in reader]
        print(len(rows), type(len(rows)), subsample_percentage*len(rows), math.floor(subsample_percentage*len(rows)))
        random_rows = random.sample(rows, math.floor(subsample_percentage*len(rows)))

        for row in random_rows:
            imageFilenames.append(row[4])
            imageClasses.append(float(row[3]))
    if shuffle_data==1:
        concatFilenamesClasses = list(zip(imageFilenames, imageClasses))
        random.shuffle(concatFilenamesClasses)
        imageFilenames, imageClasses = zip(*concatFilenamesClasses)
    return imageFilenames, imageClasses

# read train.txt (shuffle) and test.txt
trainImageFilenames, trainImageClasses = readInputFileName(trainInputFilename, 1)
valImageFilenames, valImageClasses = readInputFileName(valInputFilename, 0)
testImageFilenames, testImageClasses = readInputFileName(testInputFilename, 0)

# import numpy and tables to build the hdf5 file
import numpy as np
import tables
data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
img_dtype = tables.UInt8Atom()

# Theano and Tensorflow organized the data differently
if data_order == 'th':
    data_shape = (0, 3, 224, 224)
elif data_order == 'tf':
    data_shape = (0, 224, 224, 3)

# open the specified hdf5 file and create earrays to store the images and train_mean
hdf5_file = tables.open_file(hdf5_path, mode='w')
train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)
test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)
mean_storage = hdf5_file.create_earray(hdf5_file.root, 'train_mean', img_dtype, shape=data_shape)

# create the arrays for the outcome labels
hdf5_file.create_array(hdf5_file.root, 'train_labels', trainImageClasses)
hdf5_file.create_array(hdf5_file.root, 'val_labels', valImageClasses)
hdf5_file.create_array(hdf5_file.root, 'test_labels', testImageClasses)


# import cv2 for image reading
import cv2

# initialize mean to store the mean value in the training set
mean = np.zeros(data_shape[1:], np.float32)


# read all training images into train_storage
for i in range(len(trainImageFilenames)):
    if i % 1000 == 0 and i > 1:
        print('Processed training data: {}/{}'.format(i, len(trainImageFilenames)))
    addr = trainImageFilenames[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    train_storage.append(img[None])
    mean += img / float(len(trainImageClasses))


# read all validation images into val_storage
for i in range(len(valImageFilenames)):
    if i % 1000 == 0 and i > 1:
        print('Processed validation data: {}/{}'.format(i, len(valImageFilenames)))
    addr = valImageFilenames[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    val_storage.append(img[None])

# read all validation images into test_storage
for i in range(len(testImageFilenames)):
    if i % 1000 == 0 and i > 1:
        print('Processed test data: {}/{}'.format(i, len(testImageFilenames)))
    addr = testImageFilenames[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    test_storage.append(img[None])

# save the mean of the training set
mean_storage.append(mean[None])

# close the hdf5 file
hdf5_file.close()


# save the shuffled order of training images
with open('trainingShuffled.txt', 'w') as f:
    for i in range(len(trainImageFilenames)):
        _=f.write("%s %s\n" % (trainImageFilenames[i], trainImageClasses[i]))
