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
    
def pointnet(coor_shape):
    end_points = {}
    '''
    Pointnet Architecture
    '''
    # input_Transformation_net
    input_points =Input(shape=coor_shape) #(?, 165888, 3)
    num_points=coor_shape[0]
    print(input_points)
    
    input_image=Lambda(expand_dim)(input_points) #(?, 165888, 3, 1)
    #print(input_image.shape)
    x = Convolution2D(64, (1,3), activation='relu')(input_image)
    x = BatchNormalization()(x)
    #print(x.shape) #(?, 165888, 1, 64)
    x = Convolution2D(128, (1,1), activation='relu')(x)
    x = BatchNormalization()(x)
    #print(x.shape) #(?, 165888, 1, 128)
    x = Convolution2D(1024, (1,1), activation='relu')(x)
    x = BatchNormalization()(x)
    #print(x.shape) #(?, 165888, 1, 1024)

    x = MaxPooling2D((num_points,1))(x)
    #print(x.shape) #(?, 1, 1, 1024)
    x=  Flatten()(x)
    #(?,  1024)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    #print(x.shape) #(?, 512)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    #print(x.shape)
    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    #print(x.shape) #(?, 9)
    input_T = Reshape((3, 3))(x)
    transform=input_T
    #print(input_T.shape)#(?,3,3)


    # forward net
    point_3 = Lambda(mat_mul, arguments={'B': input_T})(input_points)
    #print(point_3.shape) #(?, 165888, 3)
    g=Lambda(expand_dim)(point_3) #(?, 165888, 3, 1)
    #print(g.shape)  #(?, 165888, 3, 1)
    g = Convolution2D(64, (1,3), activation='relu')(g)
    g = BatchNormalization()(g)
    #print(g.shape)  #(?, 165888, 1, 64)
    g = Convolution2D(64, (1,1), activation='relu')(g)
    g = BatchNormalization()(g)
    #print(g.shape) #(?, 165888, 1, 64)

    # feature transformation net
    f = Convolution2D(64, (1,1), activation='relu')(g)
    f = BatchNormalization()(f)
    #print(f.shape) #(?, 165888, 1, 64)
    f = Convolution2D(128,(1,1), activation='relu')(f)
    f = BatchNormalization()(f)
    #print(f.shape) #(?, 165888, 1, 128)
    f = Convolution2D(1024, (1,1), activation='relu')(f)
    f = BatchNormalization()(f)
    #print(f.shape) #(?, 165888, 1, 1024)
    f = MaxPooling2D((num_points,1))(f)
    f=  Flatten()(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)
    transform=feature_T
    end_points['transform']=transform
    #print(feature_T.shape) #(?, 64, 64)

    # forward net
    #print(g.shape) #(?, 165888, 1, 64)
    squeeze=Lambda(quee_dim)(g)
    #print(squeeze.shape) #(?, 165888, 64)
    g = Lambda(mat_mul, arguments={'B': feature_T})(squeeze)
    #print(g.shape) #(?, 165888, 64)
    #seg_part1 = g
    seg_part1=Lambda(expand_dimA)(g) #(?, 165888,1, 64)
    #print(seg_part1.shape)#(?, 165888,1, 64)
    h=Lambda(expand_dimA)(g)    #(?, 165888,1, 64)
    #print(h.shape)#(?, 165888,1, 64)
    g = Convolution2D(64, (1,1), activation='relu')(h)
    g = BatchNormalization()(g)
    g = Convolution2D(128,(1,1), activation='relu')(g)
    g = BatchNormalization()(g)
    #print(g.shape) ##(?, 165888,1, 128)
    g = Convolution2D(1024,(1,1), activation='relu')(g)
    g = BatchNormalization()(g)
    #print(g.shape) ##(?, 165888,1, 1024)
    # global_feature
    global_feature = MaxPooling2D((num_points,1))(g) 
    print(global_feature.shape)#(?, 1,1, 1024)
    global_feature = Lambda(exp_dim, arguments={'num_points':num_points})(global_feature)
    print(global_feature.shape) #(?, 165888, 1, 1024)
    # point_net_seg
    c = concatenate([seg_part1, global_feature])
    print(c.shape) #(?, 165888, 1, 1088)
    
    c = Convolution2D(512, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    print(c.shape) #(?, 165888, 1, 512)
    c = Convolution2D(256, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    print(c.shape)#(?, 165888, 1, 512)
    c = Convolution2D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Convolution2D(128, 1, activation='relu')(c)
    c = BatchNormalization()(c)
    print(c.shape)#(?, 165888, 1, 512)
    c= Convolution2D(2, 1, activation='softmax')(c)
    prediction=Lambda(quee_dim)(c)
    '''
    end of pointnet
    '''
    #print(end_points)
    #return prediction,end_points
    # define model
    model = Model(inputs=input_points, outputs=prediction)
    return model,end_points
