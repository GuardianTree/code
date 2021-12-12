from keras.models import Model
from keras.layers import *
from keras.initializers import orthogonal, constant, he_normal
from keras import regularizers
import numpy as np
import os
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Reshape
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Lambda, concatenate
# from keras.utils import np_utils
import h5py


def mat_mul(A, B):
    return tf.matmul(A, B)


def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1, 1])


def expand_dim(point_cloud):
    return tf.expand_dims(point_cloud, -1)


def expand_dimA(point_cloud):
    return tf.expand_dims(point_cloud, axis=[2])


def quee_dim(point_cloud):
    return tf.squeeze(point_cloud, axis=[2])
def conv(data,f,m,momen):
    x=MaxPool1D(m)(data)
    x = Conv1D(f, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1, momentum=momen)(x)
    x = Activation('relu')(x)
    return x
def con_block(data,f,k,dr,momen):
    x = Conv1D(f, 1, padding='same', kernel_initializer='he_normal')(data)
    x = BatchNormalization(axis=-1, momentum=momen)(x)
    x_a = Activation('relu')(x)
    x = Conv1D(f, k, padding='same', kernel_initializer='he_normal')(x_a)
    x = BatchNormalization(axis=-1, momentum=momen)(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)
    x = Conv1D(f, k, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1, momentum=momen)(x)
    x = Activation('relu')(x)
    con_add= Add()([x_a,x])
    return con_add
def pointnet(input_points, coor_shape):
    end_points = {}
    '''
    Pointnet Architecture
    '''
    # input_Transformation_net
    # input_points = Input(shape=coor_shape)  # (?, 165888, 3)
    num_points = coor_shape[0]
    print(input_points)
    # (B*N*3)
    # forward net
    g = Lambda(expand_dim)(input_points)  # (?, 165888, 3, 1)
    print(g.shape)  # (?, 165888, 3, 1)
    g = Convolution2D(64, (1, 3), activation='relu')(g)
    g = BatchNormalization()(g)
    print(g.shape)  # (?, 165888, 1, 64)
    g = Convolution2D(64, (1, 1), activation='relu')(g)
    g = BatchNormalization()(g)
    print(g.shape)  # (?, 165888, 1, 64)
    seg_part1 = g
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
    return prediction  # (?,165888,512)


def unet(T_1, T_2, T_3, T_4, T_5):


    f = 32
    k = 3
    dr = 0.3
    momen = 0.6
    '''downsample_T1'''
    # 576x576

    conv1_1 = Conv1D(32, 1, padding='same', kernel_initializer='he_normal')(T_1)
    batc1_1 = BatchNormalization(axis=-1, momentum=0.6)(conv1_1)
    acti1_1 = Activation('relu')(batc1_1)
    #drop1 = Dropout(0.25)(acti1)
    conv1_2 = Conv1D(32, 1, padding='same', kernel_initializer='he_normal')(T_2)
    batc1_2 = BatchNormalization(axis=-1, momentum=0.6)(conv1_2)
    acti1_2 = Activation('relu')(batc1_2)

    conv1_3 = Conv1D(32, 1, padding='same', kernel_initializer='he_normal')(T_3)
    batc1_3 = BatchNormalization(axis=-1, momentum=0.6)(conv1_3)
    acti1_3 = Activation('relu')(batc1_3)

    conv1_4 = Conv1D(32, 1, padding='same', kernel_initializer='he_normal')(T_4)
    batc1_4 = BatchNormalization(axis=-1, momentum=0.6)(conv1_4)
    acti1_4 = Activation('relu')(batc1_4)

    conv1_5 = Conv1D(32, 1, padding='same', kernel_initializer='he_normal')(T_5)
    batc1_5 = BatchNormalization(axis=-1, momentum=0.6)(conv1_5)
    acti1_5 = Activation('relu')(batc1_5)

    con = Concatenate(axis=-1)([acti1_1, acti1_2,acti1_3,acti1_4,acti1_5])
    T2 = conv(con, f, 2, momen)
    T3 = conv(con, f, 4, momen)
    T4 = conv(con, f, 8, momen)
    T5 = conv(con, f, 16, momen)
    T6 = conv(con, f, 32, momen)
    

    '''downsample_T1'''
    # 1
    conv1 = con_block(con, f * 1, k, dr, momen)
    maxp1 = MaxPool1D(2)(conv1)  # 1 4096
    # 2
    concat2 = Concatenate(axis=-1)([maxp1, T2])
    conv2 = con_block(concat2, f * 2, k, dr, momen)
    maxp2 = MaxPool1D(2)(conv2)  # 2 2048
    # 3
    concat3 = Concatenate(axis=-1)([maxp2, T3])
    conv3 = con_block(concat3, f * 4, k, dr, momen)
    maxp3 = MaxPool1D(2)(conv3)  # 3 1024
    # 4
    concat4 = Concatenate(axis=-1)([maxp3, T4])
    conv4 = con_block(concat4, f * 8, k, dr, momen)
    maxp4 = MaxPool1D(2)(conv4)  # 4 512
    # 5
    concat5 = Concatenate(axis=-1)([maxp4, T5])
    conv5 = con_block(concat5, f * 16, k, dr, momen)
    maxp5 = MaxPool1D(2)(conv5)  # 5 256
    # 6
    concat6 = Concatenate(axis=-1)([maxp5, T6])
    conv6 = con_block(concat6, f * 16, k, dr, momen)
    '''upsample_T1'''
    # 5'
    upsa_5 = UpSampling1D(2)(conv6)  # 512
    merg_5 = Concatenate(axis=-1)([conv5, upsa_5])
    conv_5 = con_block(merg_5, f * 16, k, dr, momen)

    # 4'
    upsa_4 = UpSampling1D(2)(conv_5)  # 1024
    merg_4 = Concatenate(axis=-1)([conv4, upsa_4])
    conv_4 = con_block(merg_4, f * 8, k, dr, momen)

    # 3'
    upsa_3 = UpSampling1D(2)(conv_4)  # 2048
    merg_3 = Concatenate(axis=-1)([conv3, upsa_3])
    conv_3 = con_block(merg_3, f * 4, k, dr, momen)

    # 2'
    upsa_2 = UpSampling1D(2)(conv_3)  # 4096
    merg_2 = Concatenate(axis=-1)([conv2, upsa_2])
    conv_2 = con_block(merg_2, f * 2, k, dr, momen)

    # 1'
    upsa_1 = UpSampling1D(2)(conv_2)  # 8192
    merg_1 = Concatenate(axis=-1)([conv1, upsa_1])
    conv_1 = con_block(merg_1, f * 1, k, dr, momen)
    return conv_1


def concat_net(T1_shape, T2_shape, T3_shape, T4_shape, T5_shape, coor_shape):
    input_points = Input(shape=coor_shape)
    P = pointnet(input_points, coor_shape)

    T1 = Input(shape=T1_shape)
    T2 = Input(shape=T2_shape)
    T3 = Input(shape=T3_shape)
    T4 = Input(shape=T4_shape)
    T5 = Input(shape=T5_shape)
    U = unet(T1, T2, T3, T4, T5)
    con = Concatenate(axis=-1)([U, P])
    convol = Conv1D(2, 1)(con)
    acti = Activation('softmax')(convol)

    model = Model(inputs=[T1, T2, T3, T4, T5, input_points], outputs=acti)
    return model
