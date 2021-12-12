from keras.layers import Input, BatchNormalization, MaxPool2D, Conv2D, UpSampling2D, Concatenate, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import *
import tensorflow as tf

def con_block(data,f,k,dr,momen):
    x = Conv2D(f, (k,1), padding='same', kernel_initializer='he_normal')(data)
    x = BatchNormalization(axis=-1, momentum=momen)(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)
    x = Conv2D(f, (k,1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1, momentum=momen)(x)
    x = Activation('relu')(x)
    return x
def quee_dim(point_cloud):
    return tf.squeeze(point_cloud, axis=[2])

def unet(T1_shape):
    T1 = Input(shape=T1_shape)


    f=32
    k=3
    dr=0.3
    momen=0.6

    '''downsample_T1'''


    '''downsample_T1'''
    # 1
    conv1 = con_block(T1, f * 1, k, dr, momen)
    maxp1 = MaxPool2D((2,1))(conv1)  # 1 4096
    # 2
    conv2 = con_block(maxp1, f * 2, k, dr, momen)
    maxp2 = MaxPool2D((2,1))(conv2)  # 2 2048
    # 3
    conv3 = con_block(maxp2, f * 4, k, dr, momen)
    maxp3 = MaxPool2D((2,1))(conv3)  # 3 1024
    # 4
    conv4 = con_block(maxp3, f * 8, k, dr, momen)
    maxp4 = MaxPool2D((2,1))(conv4)  # 4 512
    # 5
    conv5 = con_block(maxp4, f * 16, k, dr, momen)
    maxp5 = MaxPool2D((2,1))(conv5)  # 5 256
    # 6
    conv6 = con_block(maxp5, f * 16, k, dr, momen)

    '''upsample_T1'''
    # 5'
    upsa_5 = UpSampling2D((2,1))(conv6)  # 512
    merg_5 = Concatenate(axis=-1)([conv5, upsa_5])
    conv_5 = con_block(merg_5, f * 16, k, dr, momen)

    # 4'
    upsa_4 = UpSampling2D((2,1))(conv_5)  # 1024
    merg_4 = Concatenate(axis=-1)([conv4, upsa_4])
    conv_4 = con_block(merg_4, f * 8, k, dr, momen)

    # 3'
    upsa_3 = UpSampling2D((2,1))(conv_4)  # 2048
    merg_3 = Concatenate(axis=-1)([conv3, upsa_3])
    conv_3 = con_block(merg_3, f * 4, k, dr, momen)

    # 2'
    upsa_2 = UpSampling2D((2,1))(conv_3)  # 4096
    merg_2 = Concatenate(axis=-1)([conv2, upsa_2])
    conv_2 = con_block(merg_2, f * 2, k, dr, momen)

    # 1'
    upsa_1 = UpSampling2D((2,1))(conv_2)  # 8192
    merg_1 = Concatenate(axis=-1)([conv1, upsa_1])
    conv_1 = con_block(merg_1, f * 1, k, dr, momen)

    maxp = MaxPool2D((1,2))(conv_1)  # 5 256
    quee = Lambda(quee_dim)(maxp)
    convol = Conv1D(2, 1, padding='same')(quee)
    acti = Activation('softmax')(convol)

    model = Model(inputs=T1, outputs=acti)
    return model


