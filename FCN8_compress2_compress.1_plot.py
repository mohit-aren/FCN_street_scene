import numpy as np
import pandas as pd

import json
import sys
from PIL import Image, ImageOps

#from skimage.io import imread
#from matplotlib import pyplot as plt
import random

import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] ='mode=FAST_RUN,device=cpu'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu0, floatX=float32, optimizer=fast_compile'

from keras import models
from keras.optimizers import SGD
from keras.layers import Input, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.applications import imagenet_utils

path = 'CamSeq01/'
img_w = 960
img_h = 736
n_labels = 13
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]


n_train1 = 10
n_test = 11
n_val = 10

def label_map(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            label_map[r, c, labels[r][c]] = 1
    return label_map

def label_map1(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            #print(labels[r][c])
            if(labels[r][c][0] == Sky[0] and labels[r][c][1] == Sky[1] and labels[r][c][2] == Sky[2]):
                label_map[r, c, 0] = 1
            elif(labels[r][c][0] == Building[0] and labels[r][c][1] == Building[1] and labels[r][c][2] == Building[2]):
                label_map[r, c, 1] = 1
            elif(labels[r][c][0] == Pole[0] and labels[r][c][1] == Pole[1] and labels[r][c][2] == Pole[2]):
                label_map[r, c, 2] = 1
            elif(labels[r][c][0] == Road_marking[0] and labels[r][c][1] == Road_marking[1] and labels[r][c][2] == Road_marking[2]):
                label_map[r, c, 3] = 1
            elif(labels[r][c][0] == Road[0] and labels[r][c][1] == Road[1] and labels[r][c][2] == Road[2]):
                label_map[r, c, 4] = 1
            elif(labels[r][c][0] == Pavement[0] and labels[r][c][1] == Pavement[1] and labels[r][c][2] == Pavement[2]):
                label_map[r, c, 5] = 1
            elif(labels[r][c][0] == Tree[0] and labels[r][c][1] == Tree[1] and labels[r][c][2] == Tree[2]):
                label_map[r, c, 6] = 1
            elif(labels[r][c][0] == SignSymbol[0] and labels[r][c][1] == SignSymbol[1] and labels[r][c][2] == SignSymbol[2]):
                label_map[r, c, 7] = 1
            elif(labels[r][c][0] == Fence[0] and labels[r][c][1] == Fence[1] and labels[r][c][2] == Fence[2]):
                label_map[r, c, 8] = 1
            elif(labels[r][c][0] == Car[0] and labels[r][c][1] == Car[1] and labels[r][c][2] == Car[2]):
                label_map[r, c, 9] = 1
            elif(labels[r][c][0] == Pedestrian[0] and labels[r][c][1] == Pedestrian[1] and labels[r][c][2] == Pedestrian[2]):
                label_map[r, c, 10] = 1
            elif(labels[r][c][0] == Bicyclist[0] and labels[r][c][1] == Bicyclist[1] and labels[r][c][2] == Bicyclist[2]):
                label_map[r, c, 11] = 1
            elif(labels[r][c][0] == Unlabelled[0] and labels[r][c][1] == Unlabelled[1] and labels[r][c][2] == Unlabelled[2]):
                label_map[r, c, 12] = 1
    return label_map


import os

def prep_data1(mode):
    assert mode in {'test', 'train1', 'val'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    
    folder_path = path + mode

    images_path = os.listdir(folder_path)

    main_files = []
    truth_files = []
    
    for n, image in enumerate(images_path):
        src = os.path.join(folder_path, image)
        if(src.find('_L.png') == -1):
            main_files.append(src)
        else:
            truth_files.append(src)
    n = n_train1 if mode == 'train1' else n_test
    
    if(mode == 'val'):
        n = n_val
        
    index = 0
    for filename in main_files:
        print(filename)
        
        truth_file = filename.split('.png')
        
        tfile = truth_file[0] + '_L.png'
        
        print(tfile)
        if(filename == ""):
            break
        
        img1 = Image.open(filename)
        new_size = tuple([960, 720])
        
        # create a new image and paste the resized on it
        
        new_im = Image.new("RGB", (960, 736))
        new_im.paste(img1, ((960-new_size[0])//2,
                            (736-new_size[1])//2))


        img2 = Image.open(tfile)
        new_size = tuple([960, 720])
        
        # create a new image and paste the resized on it
        
        new_im1 = Image.new("RGB", (960, 736))
        new_im1.paste(img2, ((960-new_size[0])//2,
                            (736-new_size[1])//2))


        index += 1
        # create a new image and paste the resized on it
        

        #img, gt = [imread(path + mode + '/' + filename + '.png')], imread(path + mode + '-colormap/' + filename + '.png')
        
        img, gt = [np.array(new_im,dtype=np.uint8)], np.array(new_im1,dtype=np.uint8)
        data.append(np.reshape(img,(960,736,3)))
        label.append(label_map1(gt))
        sys.stdout.write('\r')
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    print( mode + ': OK')
    print( '\tshapes: {}, {}'.format(data.shape, label.shape))
    print( '\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    print( '\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

    return data, label




def prep_data(mode):
    assert mode in {'test', 'train'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    df = pd.read_csv(path + mode + '.csv')
    n = n_train if mode == 'train' else n_test
    for i, item in df.iterrows():
        if i >= n:
            break
        img, gt = [imread(path + item[0])], np.clip(imread(path + item[1]), 0, 1)
        data.append(np.reshape(img,(256,256,1)))
        label.append(label_map(gt))
        sys.stdout.write('\r')
        sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n)))
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    print( mode + ': OK')
    print( '\tshapes: {}, {}'.format(data.shape, label.shape))
    print( '\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    print( '\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

    return data, label

"""
def plot_results(output):
    gt = []
    df = pd.read_csv(path + 'test.csv')
    for i, item in df.iterrows():
        gt.append(np.clip(imread(path + item[1]), 0, 1))

    plt.figure(figsize=(15, 2 * n_test))
    for i, item in df.iterrows():
        plt.subplot(n_test, 4, 4 * i + 1)
        plt.title('Ground Truth')
        plt.axis('off')
        gt = imread(path + item[1])
        plt.imshow(np.clip(gt, 0, 1))

        plt.subplot(n_test, 4, 4 * i + 2)
        plt.title('Prediction')
        plt.axis('off')
        labeled = np.argmax(output[i], axis=-1)
        plt.imshow(labeled)

        plt.subplot(n_test, 4, 4 * i + 3)
        plt.title('Heat map')
        plt.axis('off')
        plt.imshow(output[i][:, :, 1])

        plt.subplot(n_test, 4, 4 * i + 4)
        plt.title('Comparison')
        plt.axis('off')
        rgb = np.empty((img_h, img_w, 3))
        rgb[:, :, 0] = labeled
        rgb[:, :, 1] = imread(path + item[0])
        rgb[:, :, 2] = gt
        plt.imshow(rgb)

    plt.savefig('result.png')
    plt.show()
"""

#########################################################################################################
"""
def SegNet(input_shape=(960, 736, 3), classes=13):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Convolution2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Convolution2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Convolution2D(classes, 1, 1, border_mode="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model
"""

from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
#from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


import numpy as np
#import matplotlib.pyplot as plt
#from pylab import *
import os
import sys
#from keras_contrib.applications import densenet
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf

from utils.get_weights_path import *
from utils.basics import *
from utils.resnet_helpers import *
from utils.BilinearUpSampling import *

from keras.models import Sequential,Model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Deconvolution2D, Cropping2D
from keras.layers import Input, Add, Dropout, Permute, add
from scipy.io import loadmat

# Function to create to a series of CONV layers followed by Max pooling layer
def Convblock(channel_dimension, block_no, no_of_convs) :
    Layers = []
    for i in range(no_of_convs) :
        
        Conv_name = "conv"+str(block_no)+"_"+str(i+1)
        
        # A constant kernel size of 3*3 is used for all convolutions
        Layers.append(Convolution2D(channel_dimension,kernel_size = (3,3),padding = "same",activation = "relu",name = Conv_name))
    
    Max_pooling_name = "pool"+str(block_no)
    
    #Addding max pooling layer
    Layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = Max_pooling_name))
    
    return Layers

def FCN_8_helper(image_size):
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = image_size))
    
    for l in Convblock(64,1,2) :
        model.add(l)
    
    for l in Convblock(128,2,2):
        model.add(l)
    
    for l in Convblock(256,3,3):
        model.add(l)
    
    for l in Convblock(512,4,3):
        model.add(l)
    
    for l in Convblock(512,5,3):
        model.add(l)
        
    model.add(Convolution2D(4096,kernel_size=(7,7),padding = "same",activation = "relu",name = "fc6"))
      
    #Replacing fully connnected layers of VGG Net using convolutions
    model.add(Convolution2D(4096,kernel_size=(1,1),padding = "same",activation = "relu",name = "fc7"))
    
    # Gives the classifications scores for each of the 21 classes including background
    model.add(Convolution2D(13,kernel_size=(1,1),padding="same",activation="relu",name = "score_fr"))
    
    Conv_size = model.layers[-1].output_shape[2] #16 if image size if 512
    #print(Conv_size)
    
    model.add(Deconvolution2D(13,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "score2"))
    
    # O = ((I-K+2*P)/Stride)+1 
    # O = Output dimesnion after convolution
    # I = Input dimnesion
    # K = kernel Size
    # P = Padding
    
    # I = (O-1)*Stride + K 
    Deconv_size = model.layers[-1].output_shape[2] #34 if image size is 512*512
    
    #print(Deconv_size)
    # 2 if image size is 512*512
    Extra = (Deconv_size - 2*Conv_size)
    
    #print(Extra)
    
    #Cropping to get correct size
    model.add(Cropping2D(cropping=((0,Extra),(0,Extra))))
    
    return model

def FCN_8(image_size=(960, 736, 3)):
    fcn_8 = FCN_8_helper(image_size)
    #Calculating conv size after the sequential block
    #32 if image size is 512*512
    Conv_size = fcn_8.layers[-1].output_shape[2] 
    
    #Conv to be applied on Pool4
    skip_con1 = Convolution2D(13,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool4")
    
    #Addig skip connection which takes adds the output of Max pooling layer 4 to current layer
    Summed = add(inputs = [skip_con1(fcn_8.layers[14].output),fcn_8.layers[-1].output])
    
    #Upsampling output of first skip connection
    x = Deconvolution2D(13,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "score4")(Summed)
    x = Cropping2D(cropping=((0,2),(0,2)))(x)
    
    
    #Conv to be applied to pool3
    skip_con2 = Convolution2D(13,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool3")
    
    #Adding skip connection which takes output og Max pooling layer 3 to current layer
    Summed = add(inputs = [skip_con2(fcn_8.layers[10].output),x])
    
    #Final Up convolution which restores the original image size
    Up = Deconvolution2D(13,kernel_size=(16,16),strides = (8,8),
                         padding = "valid",activation = None,name = "upsample")(Summed)
    
    #Cropping the extra part obtained due to transpose convolution
    final = Cropping2D(cropping = ((0,8),(0,8)))(Up)
    x = Reshape((960*736, 13))(final)
    x = Activation("softmax")(x)

    
    return Model(fcn_8.input, x)

def Convblock_compress(channel_dimension, block_no, no_of_convs) :
    Layers = []
    for i in range(no_of_convs) :
        
        Conv_name = "conv"+str(block_no)+"_"+str(i+1)
        
        # A constant kernel size of 3*3 is used for all convolutions
        Layers.append(Convolution2D(channel_dimension,kernel_size = (3,3),padding = "same",activation = "relu",name = Conv_name))
    
    #Max_pooling_name = "pool"+str(block_no)
    
    #Addding max pooling layer
    #Layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = Max_pooling_name))
    
    return Layers


def FCN_8_helper_compress(layer1_filters, layer2_filters, layer3_filters, layer4_filters, layer5_filters, layer6_filters, layer7_filters, layer8_filters,layer9_filters, layer10_filters, layer11_filters, layer12_filters, layer13_filters, layer14_filters, layer15_filters, image_size):
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = image_size))
    
    """
    for l in Convblock(64,1,2) :
        model.add(l)
    
    for l in Convblock(128,2,2):
        model.add(l)
    
    for l in Convblock(256,3,3):
        model.add(l)
    
    for l in Convblock(512,4,3):
        model.add(l)
    
    for l in Convblock(512,5,3):
        model.add(l)
    """
    for l in Convblock_compress(layer1_filters,1,1) :
        model.add(l)
    for l in Convblock_compress(layer2_filters,2,1) :
        Max_pooling_name = "pool"+str(1)
    
        model.add(l)
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = Max_pooling_name))
    for l in Convblock_compress(layer3_filters,3,1) :
        model.add(l)
    for l in Convblock_compress(layer4_filters,4,1) :
        Max_pooling_name = "pool"+str(2)
    
        model.add(l)
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = Max_pooling_name))
    for l in Convblock_compress(layer5_filters,5,1) :
        model.add(l)
    for l in Convblock_compress(layer6_filters,6,1) :
        model.add(l)
    for l in Convblock_compress(layer7_filters,7,1) :
        Max_pooling_name = "pool"+str(3)
    
        model.add(l)
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = Max_pooling_name))
    for l in Convblock_compress(layer8_filters,8,1) :
        model.add(l)
    for l in Convblock_compress(layer9_filters,9,1) :
        model.add(l)
    for l in Convblock_compress(layer10_filters,10,1) :
        Max_pooling_name = "pool"+str(4)
    
        model.add(l)
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = Max_pooling_name))
    for l in Convblock_compress(layer11_filters,11,1) :
        model.add(l)
    for l in Convblock_compress(layer12_filters,12,1) :
        model.add(l)
    for l in Convblock_compress(layer13_filters,13,1) :
        Max_pooling_name = "pool"+str(5)
    
        model.add(l)
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = Max_pooling_name))
    
    
        
    model.add(Convolution2D(layer14_filters,kernel_size=(7,7),padding = "same",activation = "relu",name = "fc6"))
      
    #Replacing fully connnected layers of VGG Net using convolutions
    model.add(Convolution2D(layer15_filters,kernel_size=(1,1),padding = "same",activation = "relu",name = "fc7"))
    
    # Gives the classifications scores for each of the 21 classes including background
    model.add(Convolution2D(13,kernel_size=(1,1),padding="same",activation="relu",name = "score_fr"))
    
    Conv_size = model.layers[-1].output_shape[2] #16 if image size if 512
    #print(Conv_size)
    
    model.add(Deconvolution2D(13,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "score2"))
    
    # O = ((I-K+2*P)/Stride)+1 
    # O = Output dimesnion after convolution
    # I = Input dimnesion
    # K = kernel Size
    # P = Padding
    
    # I = (O-1)*Stride + K 
    Deconv_size = model.layers[-1].output_shape[2] #34 if image size is 512*512
    
    #print(Deconv_size)
    # 2 if image size is 512*512
    Extra = (Deconv_size - 2*Conv_size)
    
    #print(Extra)
    
    #Cropping to get correct size
    model.add(Cropping2D(cropping=((0,Extra),(0,Extra))))
    
    return model

def FCN_8_compress(layer1_filters, layer2_filters, layer3_filters, layer4_filters, layer5_filters, layer6_filters, layer7_filters, layer8_filters,layer9_filters, layer10_filters, layer11_filters, layer12_filters, layer13_filters, layer14_filters, layer15_filters, image_size=(960, 736, 3)):
    fcn_8 = FCN_8_helper_compress(layer1_filters, layer2_filters, layer3_filters, layer4_filters, layer5_filters, layer6_filters, layer7_filters, layer8_filters,layer9_filters, layer10_filters, layer11_filters, layer12_filters, layer13_filters, layer14_filters, layer15_filters, image_size)
    #Calculating conv size after the sequential block
    #32 if image size is 512*512
    Conv_size = fcn_8.layers[-1].output_shape[2] 
    
    #Conv to be applied on Pool4
    skip_con1 = Convolution2D(13,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool4")
    
    #Addig skip connection which takes adds the output of Max pooling layer 4 to current layer
    Summed = add(inputs = [skip_con1(fcn_8.layers[14].output),fcn_8.layers[-1].output])
    
    #Upsampling output of first skip connection
    x = Deconvolution2D(13,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "score4")(Summed)
    x = Cropping2D(cropping=((0,2),(0,2)))(x)
    
    
    #Conv to be applied to pool3
    skip_con2 = Convolution2D(13,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool3")
    
    #Adding skip connection which takes output og Max pooling layer 3 to current layer
    Summed = add(inputs = [skip_con2(fcn_8.layers[10].output),x])
    
    #Final Up convolution which restores the original image size
    Up = Deconvolution2D(13,kernel_size=(16,16),strides = (8,8),
                         padding = "valid",activation = None,name = "upsample")(Summed)
    
    #Cropping the extra part obtained due to transpose convolution
    final = Cropping2D(cropping = ((0,8),(0,8)))(Up)
    x = Reshape((960*736, 13))(final)
    x = Activation("softmax")(x)

    
    return Model(fcn_8.input, x)

"""
with open('model_5l.json') as model_file:
    autoencoder = models.model_from_json(model_file.read())
"""
olayer1_filters = 27
olayer2_filters = 29
olayer3_filters = 27
olayer4_filters = 25
olayer5_filters = 39
olayer6_filters = 37
olayer7_filters = 35
olayer8_filters = 72
olayer9_filters = 74
olayer10_filters = 73
olayer11_filters = 68
olayer12_filters = 63
olayer13_filters = 74
olayer14_filters = 269
olayer15_filters = 270
autoencoder = FCN_8_compress(olayer1_filters, olayer2_filters, olayer3_filters, olayer4_filters, olayer5_filters, olayer6_filters, olayer7_filters, olayer8_filters,olayer9_filters, olayer10_filters, olayer11_filters, olayer12_filters, olayer13_filters, olayer14_filters, olayer15_filters)

print('Start')
optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print( 'Compiled: OK')
autoencoder.summary()

# Train model or load weights

test_data, test_label = prep_data1('train1')

nb_epoch = 100
batch_size = 2
#history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(val_data, val_label))
autoencoder.load_weights('FCN8_pruned_weights_5.h5')

score = autoencoder.evaluate(test_data, test_label, verbose=1)
print( 'Test score:', score[0])
print( 'Test accuracy:', score[1])

output = autoencoder.predict(test_data, verbose=1)
output = output.reshape((output.shape[0], img_h, img_w, 13))
#print(output[0])
#print(test_label.shape())
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

for index in range(0,10):
    labeled = np.argmax(output[index], axis=-1)
    labeled1 = np.zeros([img_h, img_w, 3]) 
    for i in range(0,img_h):
        for j in range(0, img_w):
            if(labeled[i,j] == 0):
                labeled1[i,j] = Sky
            elif(labeled[i,j] == 1):
                labeled1[i,j] = Building
            elif(labeled[i,j] == 2):
                labeled1[i,j] = Pole
            elif(labeled[i,j] == 3):
                labeled1[i,j] = Road_marking
            elif(labeled[i,j] == 4):
                labeled1[i,j] = Road
            elif(labeled[i,j] == 5):
                labeled1[i,j] = Pavement
            elif(labeled[i,j] == 6):
                labeled1[i,j] = Tree
            elif(labeled[i,j] == 7):
                labeled1[i,j] = SignSymbol
            elif(labeled[i,j] == 8):
                labeled1[i,j] = Fence
            elif(labeled[i,j] == 9):
                labeled1[i,j] = Car
            elif(labeled[i,j] == 10):
                labeled1[i,j] = Pedestrian
            elif(labeled[i,j] == 11):
                labeled1[i,j] = Bicyclist
            elif(labeled[i,j] == 12):
                labeled1[i,j] = Unlabelled
    
    import scipy.misc
    
    
    scipy.misc.imsave('result_fcn8_c.5.'+ str(index) + '.jpg', labeled1.astype('uint8'),'jpeg')
