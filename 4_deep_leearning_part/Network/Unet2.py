from keras.models import Model
from keras.layers import *
from keras.initializers import orthogonal, constant, he_normal
from keras import regularizers 

def SE(x,out_dim):
    ratio = 16
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
def conv(X,f):
    conv = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(X)
    conv = BatchNormalization(axis=-1)(conv)
    acti = Activation('relu')(conv)
    drop=Dropout(0.2)(acti)
    conv_1 = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(drop)
    conv_1 = BatchNormalization(axis=-1)(conv_1)
    acti_1 = Activation('relu')(conv_1)
    #Se=SE(acti_1,f)
    return acti_1

def unet(T1_shape):
    T1 = Input(shape=T1_shape)

    '''downsample_T1'''
    # 576x576
    conv2 =conv(T1,32)
    acti2 =SE(conv2,32)
    maxp1 = MaxPool2D(2)(acti2) #1

    # 288x288
    conv4 = conv(maxp1,64)
    acti4 = SE(conv4,64)
    maxp2 = MaxPool2D(2)(acti4)    #2

    # 144x144
    conv6 =conv(maxp2,128)
    acti6 = SE(conv6,128)
    maxp3 = MaxPool2D(2)(acti6)    #3

    # 72x72
    conv8 = conv(maxp3,256)
    acti8 = SE(conv8,256)
    maxp4 = MaxPool2D(2)(acti8)    #4

    # 36x36
    conv10 =conv(maxp4,512)
    acti10 = SE(conv10,512)

    '''upsample_T1'''
    upsa1 = UpSampling2D(2)(acti10)#acti8
    merg1 = Concatenate(axis=-1)([acti8, upsa1])
    acti12 = conv(merg1,256)

    upsa2 = UpSampling2D(2)(acti12)
    merg2 = Concatenate(axis=-1)([acti6, upsa2])
    acti14 = conv(merg2,128)

    upsa3 = UpSampling2D(2)(acti14)
    merg3 = Concatenate(axis=-1)([acti4, upsa3])
    acti16 = conv(merg3,64)

    upsa4 = UpSampling2D(2)(acti16)
    merg4 = Concatenate(axis=-1)([acti2, upsa4])
    acti18 = conv(merg4,32)
   
   
    convol = Conv2D(2, 1,padding='same')(acti18)
    acti = Activation('softmax')(convol)
    model = Model(inputs=T1, outputs=acti)
   

    return model
