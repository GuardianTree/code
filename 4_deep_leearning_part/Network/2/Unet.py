from keras.models import Model
from keras.layers import *
from keras.initializers import orthogonal, constant, he_normal
from keras import regularizers
def conv_block(T,f):
    conv1 = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(T)
    batc1 = BatchNormalization(axis=-1, momentum=0.6)(conv1)#, momentum=0.6
    acti1 = Activation('relu')(batc1)
    drop1 = Dropout(0.25)(acti1)
    conv2 = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(drop1)
    batc2 = BatchNormalization(axis=-1, momentum=0.6)(conv2)#, momentum=0.6
    acti2 = Activation('relu')(batc2)
    drop2 = Dropout(0.25)(acti2)
    return drop2 
def unet(T1_shape):
    f=32
    T1 = Input(shape=T1_shape)
    

    '''downsample_T1'''
    # 576x576

    conv1=conv_block(T1,f)
    maxp1 = MaxPool2D(2)(conv1)    #1

    # 288x288
    conv2=conv_block(maxp1,f*2)
    maxp2 = MaxPool2D(2)(conv2)    #2

    # 144x144
    conv3=conv_block(maxp2,f*4)
    
    maxp3 = MaxPool2D(2)(conv3)    #3

    # 72x72
    
    conv4=conv_block(maxp3,f*8)
    maxp4 = MaxPool2D(2)(conv4)    #4

    # 36x36
    conv5=conv_block(maxp4,f*16)   #5

    '''upsample_T1'''

    upsa1 = UpSampling2D(2)(conv5) #4
    merg1 = Concatenate(axis=-1)([conv4, upsa1])
    conv6=conv_block(merg1,f*8)

    upsa2 = UpSampling2D(2)(conv6) #3
    merg2 = Concatenate(axis=-1)([conv3, upsa2])
    conv7=conv_block(merg2,f*4)

    upsa3 = UpSampling2D(2)(conv7) #2
    merg3 = Concatenate(axis=-1)([conv2, upsa3])
    conv8=conv_block(merg3,f*2)
    
    upsa4 = UpSampling2D(2)(conv8) #1
    merg4 = Concatenate(axis=-1)([conv1, upsa4])
    conv9=conv_block(merg4,f*1)
 
    convol = Conv2D(2, 1,padding='same')(conv9)
    acti = Activation('softmax')(convol)
    model = Model(inputs=T1, outputs=acti)
   

    return model
