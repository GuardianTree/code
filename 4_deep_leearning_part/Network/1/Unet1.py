from keras.models import Model
from keras.layers import *
from keras.initializers import orthogonal, constant, he_normal
from keras import regularizers 
def unet(T1_shape):

    T1 = Input(shape=T1_shape)
    

    '''downsample_T1'''
    # 576x576
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(T1)
    batc1 = BatchNormalization(axis=-1, momentum=0.6)(conv1)
    acti1 = Activation('relu')(batc1)
    drop1 = Dropout(0.25)(acti1)
    conv2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(drop1)
    batc2 = BatchNormalization(axis=-1, momentum=0.6)(conv2)
    acti2 = Activation('relu')(batc2)

    maxp1 = MaxPool2D(2)(acti2)    #1

    # 288x288
    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(maxp1)
    batc3 = BatchNormalization(axis=-1, momentum=0.6)(conv3)
    acti3 = Activation('relu')(batc3)
    drop3 = Dropout(0.25)(acti3)
    conv4 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(drop3)
    batc4 = BatchNormalization(axis=-1, momentum=0.6)(conv4)
    acti4 = Activation('relu')(batc4)
   
    maxp2 = MaxPool2D(2)(acti4)    #2

    # 144x144
    conv5 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(maxp2)
    batc5 = BatchNormalization(axis=-1, momentum=0.6)(conv5)
    acti5 = Activation('relu')(batc5)
    drop5 = Dropout(0.25)(acti5)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(drop5)
    batc6 = BatchNormalization(axis=-1, momentum=0.6)(conv6)
    acti6 = Activation('relu')(batc6)
    
    maxp3 = MaxPool2D(2)(acti6)    #3

    # 72x72
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(maxp3)
    batc7 = BatchNormalization(axis=-1, momentum=0.6)(conv7)
    acti7 = Activation('relu')(batc7)
    drop7 = Dropout(0.25)(acti7)
    conv8 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(drop7)
    batc8 = BatchNormalization(axis=-1, momentum=0.6)(conv8)
    acti8 = Activation('relu')(batc8)
    maxp4 = MaxPool2D(2)(acti8)    #4

    # 36x36
    conv9 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(maxp4)
    batc9 = BatchNormalization(axis=-1, momentum=0.6)(conv9)
    acti9 = Activation('relu')(batc9)
    drop9 = Dropout(0.25)(acti9)
    conv10 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(drop9)
    batc10 = BatchNormalization(axis=-1, momentum=0.6)(conv10)
    acti10 = Activation('relu')(batc10)

    '''upsample_T1'''

    upsa1 = UpSampling2D(2)(acti10)#acti8
    # print('upsam1 shape: ', upsam1.shape)
    merg1 = Concatenate(axis=-1)([acti8, upsa1])
    conv11 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merg1)
    batc11 = BatchNormalization(axis=-1, momentum=0.6)(conv11)
    acti11 = Activation('relu')(batc11)
    drop11 = Dropout(0.25)(acti11)
    conv12 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(drop11)
    batc12 = BatchNormalization(axis=-1, momentum=0.6)(conv12)
    acti12 = Activation('relu')(batc12)

    upsa2 = UpSampling2D(2)(acti12)
    merg2 = Concatenate(axis=-1)([acti6, upsa2])
    conv13 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merg2)
    batc13 = BatchNormalization(axis=-1, momentum=0.6)(conv13)
    acti13 = Activation('relu')(batc13)
    drop13 = Dropout(0.25)(acti13)
    conv14 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(drop13)
    batc14 = BatchNormalization(axis=-1, momentum=0.6)(conv14)
    acti14 = Activation('relu')(batc14)
    

    upsa3 = UpSampling2D(2)(acti14)
    merg3 = Concatenate(axis=-1)([acti4, upsa3])
    conv15 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merg3)
    batc15 = BatchNormalization(axis=-1, momentum=0.6)(conv15)
    acti15 = Activation('relu')(batc15)
    drop15 = Dropout(0.25)(acti15)
    conv16 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(drop15)
    batc16 = BatchNormalization(axis=-1, momentum=0.6)(conv16)
    acti16 = Activation('relu')(batc16)


    upsa4 = UpSampling2D(2)(acti16)
    merg4 = Concatenate(axis=-1)([acti2, upsa4])
    conv17 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merg4)
    batc17 = BatchNormalization(axis=-1, momentum=0.6)(conv17)
    acti17 = Activation('relu')(batc17)
    drop17 = Dropout(0.25)(acti17)
    conv18 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(drop17)
    batc18 = BatchNormalization(axis=-1, momentum=0.6)(conv18)
    acti18 = Activation('relu')(batc18)
   
    
    convol = Conv2D(2, 1,padding='same')(acti18)
    acti = Activation('sigmoid')(convol)

    model = Model(inputs=T1, outputs=acti)
   

    return model