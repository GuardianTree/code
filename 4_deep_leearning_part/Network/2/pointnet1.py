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
from keras.layers import Lambda, concatenate
#from keras.utils import np_utils
import h5py

def mat_mul(A, B):
    return tf.matmul(A, B)

def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])


def pointnet(coor_shape):
    '''
    Pointnet Architecture
    '''
    # input_Transformation_net
    input_points =Input(shape=coor_shape)
    num_points=coor_shape[0]
    print(num_points)
    
    x = Convolution1D(64, 1, activation='relu')(input_points)
    x = BatchNormalization()(x)
    x = Convolution1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling1D(pool_size=num_points)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    print(x.shape)
    input_T = Reshape((3, 3))(x)

    # forward net
    g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
    g = Convolution1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # feature transformation net
    f = Convolution1D(64, 1, activation='relu')(g)
    f = BatchNormalization()(f)
    f = Convolution1D(128, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Convolution1D(1024, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = MaxPooling1D(pool_size=num_points)(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)

    # forward net
    g = Lambda(mat_mul, arguments={'B': feature_T})(g)
    seg_part1 = g
    g = Convolution1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # global_feature
    global_feature = MaxPooling1D(pool_size=num_points)(g)
    global_feature = Lambda(exp_dim, arguments={'num_points':num_points})(global_feature)

    # point_net_seg
    c = concatenate([seg_part1, global_feature])
    c = Convolution1D(512, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(256, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution1D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    prediction = Convolution1D(2, 1, activation='softmax')(c)
    '''
    end of pointnet
    '''

    # define model
    model = Model(inputs=input_points, outputs=prediction)
    return model
