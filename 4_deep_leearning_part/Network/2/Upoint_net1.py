from keras.models import Model
from keras.layers import *
from keras.initializers import orthogonal, constant, he_normal
from keras import regularizers
import numpy as np
import os
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Reshape
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D,Flatten
from keras.layers import Lambda, concatenate
#from keras.utils import np_utils
import h5py
def mat_mul(A, B):
    return tf.matmul(A, B)

def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1,1])
def expand_dim(point_cloud):
    return tf.expand_dims(point_cloud, -1)
def expand_dimA(point_cloud):
    return tf.expand_dims(point_cloud, axis=[2])
def quee_dim(point_cloud):
    return tf.squeeze(point_cloud, axis=[2])


def pointnet(input_points,coor_shape):
    end_points = {}
    '''
    Pointnet Architecture
    '''
    # input_Transformation_net
    #input_points = Input(shape=coor_shape)  # (?, 165888, 3)
    num_points = coor_shape[0]
    print(input_points)
    #(B*N*3)
    # forward net
    g = Lambda(expand_dim)(input_points)  # (?, 165888, 3, 1)
    print(g.shape)  #(?, 165888, 3, 1)
    g = Convolution2D(64, (1, 3), activation='relu')(g)
    g = BatchNormalization()(g)
    print(g.shape)  #(?, 165888, 1, 64)
    g = Convolution2D(64, (1, 1), activation='relu')(g)
    g = BatchNormalization()(g)
    print(g.shape) #(?, 165888, 1, 64)
    seg_part1=g
    seg_part = g
    h = Convolution2D(64, (1, 1), activation='relu')(seg_part)
    h = BatchNormalization()(h)
    h = Convolution2D(128, (1, 1), activation='relu')(h)
    h = BatchNormalization()(h)
    # print(g.shape) ##(?, 165888,1, 128)
    h = Convolution2D(1024, (1, 1), activation='relu')(h)
    h = BatchNormalization()(h)
    # print(g.shape) ##(?, 165888,1, 1024)
    # global_feature
    global_feature = MaxPooling2D((num_points, 1))(h)
    print(global_feature.shape)  # (?, 1,1, 1024)
    global_feature = Lambda(exp_dim, arguments={'num_points': num_points})(global_feature)
    print(global_feature.shape)  # (?, 165888, 1, 1024)
    # point_net_seg
    c = concatenate([seg_part1, global_feature])
    print(c.shape)  # (?, 165888, 1, 1088)

    c = Convolution2D(512, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    print(c.shape)  # (?, 165888, 1, 512)
    c = Convolution2D(256, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    print(c.shape)  # (?, 165888, 1, 512)
    c = Convolution2D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution2D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    print(c.shape)  # (?, 165888, 1, 512)
    prediction = Lambda(quee_dim)(c)
    return prediction  #(?,165888,512)
def unet(T1):
    #T1 = Input(shape=T1_shape)

    '''downsample_T1'''
    # 576x576
    conv1 = Conv1D(32, 3, padding='same', kernel_initializer='he_normal')(T1)
    batc1 = BatchNormalization(axis=-1, momentum=0.6)(conv1)
    acti1 = Activation('relu')(batc1)
    drop1 = Dropout(0.3)(acti1)
    conv2 = Conv1D(32, 3, padding='same', kernel_initializer='he_normal')(drop1)
    batc2 = BatchNormalization(axis=-1, momentum=0.6)(conv2)
    acti2 = Activation('relu')(batc2)

    maxp1 = MaxPool1D(2)(acti2)  # 1

    # 288x288
    conv3 = Conv1D(64, 3, padding='same', kernel_initializer='he_normal')(maxp1)
    batc3 = BatchNormalization(axis=-1, momentum=0.6)(conv3)
    acti3 = Activation('relu')(batc3)
    drop3 = Dropout(0.3)(acti3)
    conv4 = Conv1D(64, 3, padding='same', kernel_initializer='he_normal')(drop3)
    batc4 = BatchNormalization(axis=-1, momentum=0.6)(conv4)
    acti4 = Activation('relu')(batc4)

    maxp2 = MaxPool1D(2)(acti4)  # 2

    # 144x144
    conv5 = Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(maxp2)
    batc5 = BatchNormalization(axis=-1, momentum=0.6)(conv5)
    acti5 = Activation('relu')(batc5)
    drop5 = Dropout(0.3)(acti5)
    conv6 = Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(drop5)
    batc6 = BatchNormalization(axis=-1, momentum=0.6)(conv6)
    acti6 = Activation('relu')(batc6)

    maxp3 = MaxPool1D(2)(acti6)  # 3

    # 72x72
    conv7 = Conv1D(256, 3, padding='same', kernel_initializer='he_normal')(maxp3)
    batc7 = BatchNormalization(axis=-1, momentum=0.6)(conv7)
    acti7 = Activation('relu')(batc7)
    drop7 = Dropout(0.3)(acti7)
    conv8 = Conv1D(256, 3, padding='same', kernel_initializer='he_normal')(drop7)
    batc8 = BatchNormalization(axis=-1, momentum=0.6)(conv8)
    acti8 = Activation('relu')(batc8)
    maxp4 = MaxPool1D(2)(acti8)  # 4

    # 36x36
    conv9 = Conv1D(512, 3, padding='same', kernel_initializer='he_normal')(maxp4)
    batc9 = BatchNormalization(axis=-1, momentum=0.6)(conv9)
    acti9 = Activation('relu')(batc9)
    drop9 = Dropout(0.3)(acti9)
    conv10 = Conv1D(512, 3, padding='same', kernel_initializer='he_normal')(drop9)
    batc10 = BatchNormalization(axis=-1, momentum=0.6)(conv10)
    acti10 = Activation('relu')(batc10)
    maxp5 = MaxPool1D(2)(acti10)

    conv_11 = Conv1D(512, 3, padding='same', kernel_initializer='he_normal')(maxp5)
    batc_11 = BatchNormalization(axis=-1, momentum=0.6)(conv_11)
    acti_11 = Activation('relu')(batc_11)
    drop_11 = Dropout(0.3)(acti_11)
    conv_12 = Conv1D(512, 3, padding='same', kernel_initializer='he_normal')(drop_11)
    batc_12 = BatchNormalization(axis=-1, momentum=0.6)(conv_12)
    acti_12 = Activation('relu')(batc_12)

    '''upsample_T1'''

    upsa_1 = UpSampling1D(2)(acti_12)  # acti8
    # print('upsam1 shape: ', upsam1.shape)
    merg_1 = Concatenate(axis=-1)([acti10, upsa_1])
    conv11_ = Conv1D(256, 3, padding='same', kernel_initializer='he_normal')(merg_1)
    batc11_ = BatchNormalization(axis=-1, momentum=0.6)(conv11_)
    acti11_ = Activation('relu')(batc11_)
    drop11_ = Dropout(0.3)(acti11_)
    conv12_ = Conv1D(256, 3, padding='same', kernel_initializer='he_normal')(drop11_)
    batc12_ = BatchNormalization(axis=-1, momentum=0.6)(conv12_)
    acti12_ = Activation('relu')(batc12_)

    upsa1 = UpSampling1D(2)(acti12_)  # acti8
    # print('upsam1 shape: ', upsam1.shape)
    merg1 = Concatenate(axis=-1)([acti8, upsa1])
    conv11 = Conv1D(256, 3, padding='same', kernel_initializer='he_normal')(merg1)
    batc11 = BatchNormalization(axis=-1, momentum=0.6)(conv11)
    acti11 = Activation('relu')(batc11)
    drop11 = Dropout(0.3)(acti11)
    conv12 = Conv1D(256, 3, padding='same', kernel_initializer='he_normal')(drop11)
    batc12 = BatchNormalization(axis=-1, momentum=0.6)(conv12)
    acti12 = Activation('relu')(batc12)

    upsa2 = UpSampling1D(2)(acti12)
    merg2 = Concatenate(axis=-1)([acti6, upsa2])
    conv13 = Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(merg2)
    batc13 = BatchNormalization(axis=-1, momentum=0.6)(conv13)
    acti13 = Activation('relu')(batc13)
    drop13 = Dropout(0.3)(acti13)
    conv14 = Conv1D(128, 3, padding='same', kernel_initializer='he_normal')(drop13)
    batc14 = BatchNormalization(axis=-1, momentum=0.6)(conv14)
    acti14 = Activation('relu')(batc14)

    upsa3 = UpSampling1D(2)(acti14)
    merg3 = Concatenate(axis=-1)([acti4, upsa3])
    conv15 = Conv1D(64, 3, padding='same', kernel_initializer='he_normal')(merg3)
    batc15 = BatchNormalization(axis=-1, momentum=0.6)(conv15)
    acti15 = Activation('relu')(batc15)
    drop15 = Dropout(0.3)(acti15)
    conv16 = Conv1D(64, 3, padding='same', kernel_initializer='he_normal')(drop15)
    batc16 = BatchNormalization(axis=-1, momentum=0.6)(conv16)
    acti16 = Activation('relu')(batc16)

    upsa4 = UpSampling1D(2)(acti16)
    merg4 = Concatenate(axis=-1)([acti2, upsa4])
    conv17 = Conv1D(32, 3, padding='same', kernel_initializer='he_normal')(merg4)
    batc17 = BatchNormalization(axis=-1, momentum=0.6)(conv17)
    acti17 = Activation('relu')(batc17)
    drop17 = Dropout(0.3)(acti17)
    conv18 = Conv1D(32, 3, padding='same', kernel_initializer='he_normal')(drop17)
    batc18 = BatchNormalization(axis=-1, momentum=0.6)(conv18)
    acti18 = Activation('relu')(batc18)
    return acti18  #(?,4096,32)

def concat_net(T1_shape,coor_shape):
    input_points = Input(shape=coor_shape)
    P =pointnet(input_points, coor_shape)
    T1 = Input(shape=T1_shape)
    U=unet(T1)
    con=Concatenate(axis=-1)([U, P])
    convol = Conv1D(2, 1)(con)
    acti = Activation('softmax')(convol)

    model = Model(inputs=[T1,input_points], outputs=acti)
    return model
