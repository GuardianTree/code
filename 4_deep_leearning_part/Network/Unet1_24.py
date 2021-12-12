from keras.models import Model
from keras.layers import *
from keras.initializers import orthogonal, constant, he_normal
from keras import regularizers
def SE(x,out_dim):
    ratio = 24
    #print(x.shape)
    squeeze = GlobalAveragePooling2D()(x)
    #print(squeeze.shape)
    excitation = Dense(units=out_dim // ratio)(squeeze)
    #print(out_dim // ratio)
    #print(excitation.shape)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    #print(excitation.shape)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1,out_dim))(excitation)
    #print(excitation.shape)   
    scale = multiply([x,excitation])
    #print( scale.shape)   
    return scale

def conv(x,f):
    conv_1 = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(x)
    batc_1 = BatchNormalization(axis=-1, momentum=0.6)(conv_1)
    acti_1 = Activation('relu')(batc_1)
    drop_1 = Dropout(0.2)(acti_1)
    conv_2 = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(drop_1)
    batc_2 = BatchNormalization(axis=-1, momentum=0.6)(conv_2)
    acti_2 = Activation('relu')(batc_2)
    Se=SE(acti_2,f)
    return Se

def unet(T1_shape):

    T1 = Input(shape=T1_shape)
    f=32

    '''downsample_T1'''
    # 576x576
    conv1=conv(T1,f)
    maxp1 = MaxPool2D(2)(conv1)    #1

    # 288x288
    conv2=conv(maxp1,f*2)
   
    maxp2 = MaxPool2D(2)(conv2)    #2

    # 144x144
    conv3=conv(maxp2,f*4)
    maxp3 = MaxPool2D(2)(conv3)    #3

    # 72x72
    conv4=conv(maxp3,f*8)
    maxp4 = MaxPool2D(2)(conv4)    #4

    # 36x36
    conv5=conv(maxp4,f*16)

    '''upsample_T1'''

    upsa1 = UpSampling2D(2)(conv5)#acti8
    # print('upsam1 shape: ', upsam1.shape)
    merg1 = Concatenate(axis=-1)([conv4, upsa1])
    conv_1=conv(merg1,f*8)

    upsa2 = UpSampling2D(2)(conv_1)
    merg2 = Concatenate(axis=-1)([conv3, upsa2])
    conv_2=conv(merg2,f*4)

    upsa3 = UpSampling2D(2)(conv_2)
    merg3 = Concatenate(axis=-1)([conv2, upsa3])
    conv_3=conv(merg3,f*2)


    upsa4 = UpSampling2D(2)(conv_3)
    merg4 = Concatenate(axis=-1)([conv1, upsa4])
    conv_4=conv(merg4,f)

    
    
    convol = Conv2D(2, 1,padding='same')(conv_4)
    acti = Activation('softmax')(convol)

    model = Model(inputs=T1, outputs=acti)
   

    return model
