from keras.models import Model
from keras.layers import *
from keras.initializers import orthogonal, constant, he_normal
from keras import regularizers
import keras
def Block(T,f):
    conv=Conv2D(f,3, padding='same', kernel_initializer='he_normal')(T)
    bat1 = BatchNormalization(axis=-1)(conv)#, momentum=0.6
    act1 = Activation('relu')(bat1)
    con2=Conv2D(f,3, padding='same', kernel_initializer='he_normal')(act1)
    bat2 = BatchNormalization(axis=-1)(con2)#, momentum=0.6
    act2 = Activation('relu')(bat2)
    con_add= keras.layers.Add()([T,act2])
    return con_add
def Block1(T,f):
    conv_0=Conv2D(f,1, padding='same', kernel_initializer='he_normal')(T)
    conv_0 = BatchNormalization(axis=-1)(conv_0)#, momentum=0.6
    conv_0 = Activation('relu')(conv_0)
    #drop1 = Dropout(0.3)(acti1)
    conv_1=Conv2D(f,3, padding='same', kernel_initializer='he_normal')(conv_0)
    conv_1 = BatchNormalization(axis=-1)(conv_1)#, momentum=0.6
    conv_1 = Activation('relu')(conv_1)
    conv_1 = Dropout(0.3)(conv_1)
    conv_2=Conv2D(f,1, padding='same', kernel_initializer='he_normal')(conv_1)
    conv_2 = BatchNormalization(axis=-1)(conv_2)#, momentum=0.6
    conv_2 = Activation('relu')(conv_2)
    con_add= keras.layers.Add()([conv_0,conv_2])
    return con_add
def Resunet(T1_shape):
    f=32*2
    T1 = Input(shape=T1_shape)
    

    '''downsample_T1'''
    # 576x576
    conv1=Block1(T1,f*1)
    maxp1 = MaxPool2D(2)(conv1)    #1

    # 288x288
    conv2=Block1(maxp1,f*2)
    maxp2 = MaxPool2D(2)(conv2)    #2

    # 144x144
    
    conv3=Block1(maxp2,f*4)
    maxp3 = MaxPool2D(2)(conv3)    #3

    # 72x72
    conv4=Block1(maxp3,f*8)
    maxp4 = MaxPool2D(2)(conv4)    #4

    # 36x36
    conv5=Block1(maxp4,f*16)
    

    '''upsample_T1'''

    upsa1 = UpSampling2D(2)(conv5)     #4
    merg1 = Concatenate(axis=-1)([conv4, upsa1])
    conv6=Block1(merg1,f*8)
    
    upsa2 = UpSampling2D(2)(conv6)     #3
    merg2 = Concatenate(axis=-1)([conv3, upsa2])
    conv7=Block1(merg2,f*4)
    
    upsa3 = UpSampling2D(2)(conv7)     #2
    merg3 = Concatenate(axis=-1)([conv2, upsa3])
    conv8=Block1(merg3,f*2)

    upsa4 = UpSampling2D(2)(conv8)     #1
    merg4 = Concatenate(axis=-1)([conv1, upsa4])
    conv9=Block1(merg4,f*1)
    
    convol = Conv2D(2, 1,padding='same')(conv9)
    acti = Activation('softmax')(convol)

    model = Model(inputs=T1, outputs=acti)
   

    return model
