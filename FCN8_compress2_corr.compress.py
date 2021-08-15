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


n_train = 80
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
    assert mode in {'test', 'train', 'val'}, \
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
    n = n_train if mode == 'train' else n_test
    
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
#########################################################################################################


"""
with open('model_5l.json') as model_file:
    autoencoder = models.model_from_json(model_file.read())
"""

autoencoder = FCN_8()

print('Start')
optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print( 'Compiled: OK')
autoencoder.summary()

# Train model or load weights

train_data, train_label = prep_data1('train')
val_data, val_label = prep_data1('val')
nb_epoch = 100
batch_size = 2
#history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(val_data, val_label))
#autoencoder.save_weights('model_5l_weight_fcn_CamSeq2.hdf5')

#autoencoder.load_weights('model_5l_weight_ep50.hdf5')


olayer1_filters = 64
olayer2_filters = 64
olayer3_filters = 128
olayer4_filters = 128
olayer5_filters = 256
olayer6_filters = 256
olayer7_filters = 256
olayer8_filters = 512
olayer9_filters = 512
olayer10_filters = 512
olayer11_filters = 512
olayer12_filters = 512
olayer13_filters = 512
olayer14_filters = 4096
olayer15_filters = 4096

test_data, test_label = prep_data1('test')

def enure_binary(x):
    y = []
    for indx in range(0, len(x)):
        if(x[indx] < 0.5):
            y.append(0)
        else:
            y.append(1)
            
    return y
            
            
num_dimensions = 4096
        # print final results
class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            #print(num_dimensions)
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]


for withtrain in range(0,11):
    model_final = FCN_8_compress(olayer1_filters, olayer2_filters, olayer3_filters, olayer4_filters, olayer5_filters, olayer6_filters, olayer7_filters, olayer8_filters,olayer9_filters, olayer10_filters, olayer11_filters, olayer12_filters, olayer13_filters, olayer14_filters, olayer15_filters)

    print('Start')
    optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print( 'Compiled: OK')
    model_final.summary()
    
    wt = "FCN8_pruned_weights_" + str(withtrain) + ".h5"
    model_final.load_weights(wt)

    nlayer1_filters = olayer1_filters
    nlayer2_filters = olayer2_filters
    nlayer3_filters = olayer3_filters
    nlayer4_filters = olayer4_filters
    nlayer5_filters = olayer5_filters
    nlayer6_filters = olayer6_filters
    nlayer7_filters = olayer7_filters
    nlayer8_filters = olayer8_filters
    nlayer9_filters = olayer9_filters
    nlayer10_filters = olayer10_filters
    nlayer11_filters = olayer11_filters
    nlayer12_filters = olayer12_filters
    nlayer13_filters = olayer13_filters
    nlayer14_filters = olayer14_filters
    nlayer15_filters = olayer15_filters

    ####################### 1st convolution layer with olayer1_filters filters
    print('1st convolution layer with olayer1_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[2].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer1_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[2].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer1_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer1_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer1_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A1 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer1_filters = new_num
    
    ####################### 2nd convolution layer with olayer2_filters filters
    print('2nd convolution layer with olayer2_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[3].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer2_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[3].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer2_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer2_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer2_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)

    print(func1(par1))
    A2 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer2_filters = new_num
    
    ####################### 3rd convolution layer with olayer3_filters filters
    print('3rd convolution layer with olayer3_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[5].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer3_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[5].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer3_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer3_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer3_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A3 = np.copy(par1)       
    new_num = np.sum(par1)

           
    print(new_num)
    olayer3_filters = new_num

    ####################### 4th convolution layer with olayer4_filters filters
    print('4th convolution layer with olayer4_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[6].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer4_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[6].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer4_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer4_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer4_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)

    print(func1(par1))
    A4 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer4_filters = new_num
    
    ####################### 5th convolution layer with olayer5_filters filters
    print('5th convolution layer with olayer5_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[8].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer5_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[8].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer5_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer5_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer5_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A5 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer5_filters = new_num
    
    ####################### 6th convolution layer with olayer6_filters filters
    print('6th convolution layer with olayer6_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[9].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer6_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[9].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer6_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer6_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer6_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A6 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer6_filters = new_num

    ####################### 7th convolution layer with olayer7_filters filters
    print('7th convolution layer with olayer7_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[10].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer7_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[10].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer7_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer7_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer7_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A7 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer7_filters = new_num
    
    ####################### 8th convolution layer with olayer8_filters filters
    print('8th convolution layer with olayer8_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[12].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer8_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[12].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer8_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer8_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer8_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A8 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer8_filters = new_num
    
    ####################### 9th convolution layer with olayer9_filters filters
    print('9th convolution layer with olayer9_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[13].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer9_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[13].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):


        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer9_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer9_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer9_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A9 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer9_filters = new_num
    
    ####################### 10th convolution layer with olayer10_filters filters
    print('10th convolution layer with olayer10_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[14].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer10_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[14].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):

        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer10_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer10_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer10_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A10 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer10_filters = new_num
    
    ####################### 11th convolution layer with olayer11_filters filters
    print('11th convolution layer with olayer11_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[16].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer11_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[16].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):

        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer11_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer11_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer11_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A11 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer11_filters = new_num
    
    ####################### 12th convolution layer with olayer12_filters filters
    print('12th convolution layer with olayer12_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[17].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer12_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[17].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):

        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer12_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer12_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer12_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A12 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer12_filters = new_num
    
   
    ####################### 13th convolution layer with olayer13_filters filters
    print('13th convolution layer with olayer13_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[18].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer13_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[18].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):

        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer13_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer13_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer13_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A13 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer13_filters = new_num
    
    ####################### 14th convolution layer with olayer14_filters filters
    print('14th convolution layer with olayer14_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[20].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer14_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[20].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):

        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer14_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer14_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer14_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A14 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer14_filters = new_num
    
    ####################### 15th convolution layer with olayer15_filters filters
    print('15th convolution layer with olayer15_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[21].get_weights()
    #filters1 = np.copy(filters)
    #biases1 = np.copy(biases)
    
    def func1(x):
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
        
        for i in range(0,olayer15_filters):
            if(x[i] < 0.5):
                biases1[i] = 0
                filters1[:, :, :, i] = 0
        
        model_final.layers[21].set_weights([filters1, biases1])
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        score_trial = 0.5*arr[1]+0.5*len(x)/np.sum(x)

                
        return score_trial
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):

        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        return pos_best_g

    indv = []
    bounds = []
    for k in range(0, olayer15_filters):
        indv.append(random.randint(0,1))
        bounds.append((0,1))
            
    if(olayer15_filters > 32):
        par1 = PSO(func1,indv,bounds,num_particles=15,maxiter=30)
    else:
        par1 = []
        for k in range(0, olayer15_filters):
            par1.append(1)
    
    par1 = enure_binary(par1)
    print(func1(par1))
    A15 = np.copy(par1)       
    new_num = np.sum(par1)
           
    print(new_num)
    olayer15_filters = new_num
    
    model1 = FCN_8_compress(olayer1_filters, olayer2_filters, olayer3_filters, olayer4_filters, olayer5_filters, olayer6_filters, olayer7_filters, olayer8_filters,olayer9_filters, olayer10_filters, olayer11_filters, olayer12_filters, olayer13_filters, olayer14_filters, olayer15_filters)

    print('Start')
    optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    model1.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print( 'Compiled: OK')
    model1.summary()
    
    layerr = model_final.layers[0].get_weights()
    model1.layers[0].set_weights(layerr)
    
    layerr = model_final.layers[1].get_weights()
    model1.layers[1].set_weights(layerr)
    
    model = model_final
    ######################## 1st convolution layer with 128 filters
    filters, biases = model.layers[2].get_weights()
    filters1, biases1 = model1.layers[2].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer1_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer1_filters):
        if(A1[j] == 1) :
            """
            for i1 in range (0,3):
                for j1 in range(0,3):
                    filters1[:, :, index1][:,:,j][i1][j1] = filters[:, :, i][:,:,j][i1][j1]
            """
            filters1[:, :, :, index1] = filters[:, :, :, j]
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[2].set_weights([filters1, biases1])
    
    ######################## 1st convolution layer with 256 filters
    filters, biases = model.layers[3].get_weights()
    filters1, biases1 = model1.layers[3].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer2_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer2_filters):
        if(A2[j] == 1) :
            index2 = 0
            for l in range(nlayer1_filters):
                if(A1[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[3].set_weights([filters1, biases1])
    
    ######################## 1st convolution layer with 512 filters
    filters, biases = model.layers[5].get_weights()
    filters1, biases1 = model1.layers[5].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer3_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer3_filters):
        if(A3[j] == 1) :
            index2 = 0
            for l in range(nlayer2_filters):
                if(A2[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[5].set_weights([filters1, biases1])
    
    ######################## 2nd convolution layer with 512 filters
    filters, biases = model.layers[6].get_weights()
    filters1, biases1 = model1.layers[6].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer4_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer4_filters):
        if(A4[j] == 1) :
            index2 = 0
            for l in range(nlayer3_filters):
                if(A3[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[6].set_weights([filters1, biases1])
    
    ######################## 1st dense layer with 1024 filters
    filters, biases = model.layers[8].get_weights()
    filters1, biases1 = model1.layers[8].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer5_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer5_filters):
        if(A5[j] == 1) :
            index2 = 0
            for l in range(nlayer4_filters):
                if(A4[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[8].set_weights([filters1, biases1])
    
    ######################## conv layer with 256 filters
    filters, biases = model.layers[9].get_weights()
    filters1, biases1 = model1.layers[9].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer6_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer6_filters):
        if(A6[j] == 1) :
            index2 = 0
            for l in range(nlayer5_filters):
                if(A5[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[9].set_weights([filters1, biases1])
    
    ######################## conv layer with 256 filters
    filters, biases = model.layers[10].get_weights()
    filters1, biases1 = model1.layers[10].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer7_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer7_filters):
        if(A7[j] == 1) :
            index2 = 0
            for l in range(nlayer6_filters):
                if(A6[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[10].set_weights([filters1, biases1])
    
    ######################## conv layer with 512 filters
    filters, biases = model.layers[12].get_weights()
    filters1, biases1 = model1.layers[12].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer8_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer8_filters):
        if(A8[j] == 1) :
            index2 = 0
            for l in range(nlayer7_filters):
                if(A7[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[12].set_weights([filters1, biases1])
    
    ######################## conv layer with 512 filters
    filters, biases = model.layers[13].get_weights()
    filters1, biases1 = model1.layers[13].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer9_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer9_filters):
        if(A9[j] == 1) :
            index2 = 0
            for l in range(nlayer8_filters):
                if(A8[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[13].set_weights([filters1, biases1])
    
    ######################## conv layer with 512 filters
    filters, biases = model.layers[14].get_weights()
    filters1, biases1 = model1.layers[14].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer10_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer10_filters):
        if(A10[j] == 1) :
            index2 = 0
            for l in range(nlayer9_filters):
                if(A9[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[14].set_weights([filters1, biases1])
     
    ######################## conv layer with 512 filters
    filters, biases = model.layers[16].get_weights()
    filters1, biases1 = model1.layers[16].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer11_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer11_filters):
        if(A11[j] == 1) :
            index2 = 0
            for l in range(nlayer10_filters):
                if(A10[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[16].set_weights([filters1, biases1])
    
    ######################## conv layer with 512 filters
    filters, biases = model.layers[17].get_weights()
    filters1, biases1 = model1.layers[17].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer12_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer12_filters):
        if(A12[j] == 1) :
            index2 = 0
            for l in range(nlayer11_filters):
                if(A11[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[17].set_weights([filters1, biases1])
    
    ######################## conv layer with 512 filters
    filters, biases = model.layers[18].get_weights()
    filters1, biases1 = model1.layers[18].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer13_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer13_filters):
        if(A13[j] == 1) :
            index2 = 0
            for l in range(nlayer12_filters):
                if(A12[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[18].set_weights([filters1, biases1])
    
    ######################## 2nd dense layer with 1024 filters
    filters, biases = model.layers[20].get_weights()
    filters1, biases1 = model1.layers[20].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer14_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer14_filters):
        if(A14[j] == 1) :
            index2 = 0
            for l in range(nlayer13_filters):
                if(A13[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[20].set_weights([filters1, biases1])
    
    ######################## 2nd dense layer with 1024 filters
    filters, biases = model.layers[21].get_weights()
    filters1, biases1 = model1.layers[21].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer15_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer15_filters):
        if(A15[j] == 1) :
            index2 = 0
            for l in range(nlayer14_filters):
                if(A14[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[21].set_weights([filters1, biases1])
    
    
    history = model1.fit(train_data, train_label, batch_size=batch_size, nb_epoch=50, verbose=1, validation_data=(val_data, val_label))
    
    #model1.load_weights('model_5l_weight_fcn_CamSeq2.hdf5')
    
    #test_data, test_label = prep_data1('test')
    score = model1.evaluate(test_data, test_label, verbose=1)
    print( 'Test score:', score[0])
    print( 'Test accuracy:', score[1])
    
    wt1 = "FCN8_pruned_weights_" + str(withtrain+1) + ".h5"

    model1.save_weights(wt1)
#score = autoencoder.evaluate(test_data, test_label, verbose=1)
#print( 'Test score:', score[0])
#print( 'Test accuracy:', score[1])

