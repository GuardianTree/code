import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense,Conv1D,GlobalMaxPooling1D,Dropout,Flatten,Input,Activation,BatchNormalization,Reshape
from keras.optimizers import Adam
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras import metrics
from keras.utils import to_categorical


def input_transform_net(input1):
        k=3
        #print(input1.shape)
        conv_3=Conv1D(64, kernel_size=1, strides=1, padding='valid', data_format='channels_last',
              activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros', kernel_regularizer=None)(input1)
        bn_3=BatchNormalization()(conv_3)
        conv_4=Conv1D(128, kernel_size=1, strides=1, padding='valid', data_format='channels_last',
              activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros', kernel_regularizer=None)(bn_3)
        bn_4=BatchNormalization()(conv_4)
        conv_5=Conv1D(1024, kernel_size=1, strides=1, padding='valid', data_format='channels_last',
              activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros', kernel_regularizer=None)(bn_4)
        bn_5=BatchNormalization()(conv_5)
        
        maxpool_1=GlobalMaxPooling1D()(bn_5)

        dense_1=Dense(512, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros', kernel_regularizer=None)(maxpool_1)
        bn_6=BatchNormalization()(dense_1)
        dense_2=Dense(256, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros', kernel_regularizer=None)(bn_6)
        bn_7=BatchNormalization()(dense_2)

        transform=Dense(k*k, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', 
                bias_initializer='zeros', kernel_regularizer=None)(bn_7)
        input_transform_matrix = Reshape((k,k))(transform)
        #print(input_transform_matrix.shape)

        return input_transform_matrix

def feat_transform_net(input1):
        k=64
        conv_3=Conv1D(64, kernel_size=1, strides=1, padding='valid', data_format='channels_last',
              activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros', kernel_regularizer=None)(input1)
        bn_3=BatchNormalization()(conv_3)
        conv_4=Conv1D(128, kernel_size=1, strides=1, padding='valid', data_format='channels_last',
              activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros', kernel_regularizer=None)(bn_3)
        bn_4=BatchNormalization()(conv_4)
        conv_5=Conv1D(1024, kernel_size=1, strides=1, padding='valid', data_format='channels_last',
              activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros', kernel_regularizer=None)(bn_4)
        bn_5=BatchNormalization()(conv_5)

        maxpool_1=GlobalMaxPooling1D()(bn_5)

        dense_1=Dense(512, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros', kernel_regularizer=None)(maxpool_1)
        bn_6=BatchNormalization()(dense_1)
        dense_2=Dense(256, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros', kernel_regularizer=None)(bn_6)
        bn_7=BatchNormalization()(dense_2)

        transform=Dense(k*k, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros', kernel_regularizer=None)(bn_7)

        feat_transform_matrix=Reshape((k,k))(transform)

        return feat_transform_matrix
