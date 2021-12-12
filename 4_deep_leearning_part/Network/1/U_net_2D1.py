from keras.layers import Input, BatchNormalization, MaxPool2D, Conv2D, UpSampling2D, Concatenate, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import *

def unet(T1_shape):

    T1 = Input(shape=T1_shape)
    

    '''downsample_T1'''
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(T1)
    batc1 = BatchNormalization(axis=-1)(conv1)
    acti1 = Activation('relu')(batc1)
    drop1 = Dropout(0.25)(acti1)
    conv2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(drop1)
    batc2 = BatchNormalization(axis=-1)(conv2)
    acti2 = Activation('relu')(batc2)
    drop2 = Dropout(0.25)(acti2)
    maxp1 = MaxPool2D(2)(acti2)


    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(maxp1)
    batc3 = BatchNormalization(axis=-1)(conv3)
    acti3 = Activation('relu')(batc3)
    drop3 = Dropout(0.25)(acti3)
    conv4 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(drop3)
    batc4 = BatchNormalization(axis=-1)(conv4)
    acti4 = Activation('relu')(batc4)
    drop4 = Dropout(0.25)(acti4)
    maxp2 = MaxPool2D(2)(acti4)


    conv5 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(maxp2)
    batc5 = BatchNormalization(axis=-1)(conv5)
    acti5 = Activation('relu')(batc5)
    drop5 = Dropout(0.25)(acti5)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(drop5)
    batc6 = BatchNormalization(axis=-1)(conv6)
    acti6 = Activation('relu')(batc6)
    drop6 = Dropout(0.25)(acti6)
    maxp3 = MaxPool2D(2)(acti6)

    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(maxp3)
    batc7 = BatchNormalization(axis=-1)(conv7)
    acti7 = Activation('relu')(batc7)
    drop7 = Dropout(0.25)(acti7)
    conv8 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(drop7)
    batc8 = BatchNormalization(axis=-1)(conv8)
    acti8 = Activation('relu')(batc8)
    """
    drop8 = Dropout(0.25)(acti8)
    maxp4 = MaxPool2D(2)(drop8)
    
    conv9 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(maxp4)
    batc9 = BatchNormalization(axis=-1)(conv9)
    acti9 = Activation('relu')(batc9)
    drop9 = Dropout(0.25)(acti9)
    conv10 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(drop9)
    batc10 = BatchNormalization(axis=-1)(conv10)
    acti10 = Activation('relu')(batc10)
    
    drop10 = Dropout(0.25)(acti10)
    maxp5 = MaxPool2D(2)(drop10)


    conv11 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(maxp5)
    batc11 = BatchNormalization(axis=-1)(conv11)
    acti11 = Activation('relu')(batc11)
    drop11 = Dropout(0.25)(acti11)
    conv12 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(drop11)
    batc12 = BatchNormalization(axis=-1)(conv12)
    acti12 = Activation('relu')(batc12)
    

    drop12 = Dropout(0.25)(acti12)
    maxp6 = MaxPool2D(2)(drop12)
    conv13 = Conv2D(2048, 3, padding='same', kernel_initializer='he_normal')(maxp6)
    batc13 = BatchNormalization(axis=-1)(conv13)
    acti13 = Activation('relu')(batc13)
    drop13 = Dropout(0.25)(acti13)
    conv14 = Conv2D(2048, 3, padding='same', kernel_initializer='he_normal')(drop13)
    batc14 = BatchNormalization(axis=-1)(conv14)
    acti14 = Activation('relu')(batc14)

    
    upsa_1 = UpSampling2D(2)(acti14)
    # print('upsam1 shape: ', upsam1.shape)
    merg_1 = Concatenate(axis=-1)([conv12, upsa_1])
    conv_9 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(merg_1)
    batc_9 = BatchNormalization(axis=-1)(conv_9)
    acti_9 = Activation('relu')(batc_9)
    drop_9 = Dropout(0.25)(acti_9)
    conv_10 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(drop_9)
    batc_10 = BatchNormalization(axis=-1)(conv_10)
    acti_10 = Activation('relu')(batc_10)
    drop_10 = Dropout(0.25)(acti_10)
    

    upsa_2 = UpSampling2D(2)(acti12)
    # print('upsam1 shape: ', upsam1.shape)
    merg_2 = Concatenate(axis=-1)([conv10, upsa_2])
    conv_11 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merg_2)
    batc_11 = BatchNormalization(axis=-1)(conv_11)
    acti_11 = Activation('relu')(batc_11)
    drop_11 = Dropout(0.25)(acti_11)
    conv_12 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(drop_11)
    batc_12 = BatchNormalization(axis=-1)(conv_12)
    acti_12 = Activation('relu')(batc_12)
    drop_12 = Dropout(0.25)(acti_12)
    

    upsa_3 = UpSampling2D(2)(acti10)
    # print('upsam1 shape: ', upsam1.shape)
    merg_3 = Concatenate(axis=-1)([conv8, upsa_3])
    conv_13 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merg_3)
    batc_13 = BatchNormalization(axis=-1)(conv_13)
    acti_13 = Activation('relu')(batc_13)
    drop_13 = Dropout(0.25)(acti_13)
    conv_14 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(drop_13)
    batc_14 = BatchNormalization(axis=-1)(conv_14)
    acti_14 = Activation('relu')(batc_14)
    drop_14 = Dropout(0.25)(acti_14)
    """


    upsa1 = UpSampling2D(2)(acti8)#acti8
    # print('upsam1 shape: ', upsam1.shape)
    merg1 = Concatenate(axis=-1)([conv6, upsa1])
    conv9 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merg1)
    batc9 = BatchNormalization(axis=-1)(conv9)
    acti9 = Activation('relu')(batc9)
    drop9 = Dropout(0.25)(acti9)
    conv10 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(drop9)
    batc10 = BatchNormalization(axis=-1)(conv10)
    acti10 = Activation('relu')(batc10)
    drop10 = Dropout(0.25)(acti10)



    

    upsa2 = UpSampling2D(2)(acti10)
    merg2 = Concatenate(axis=-1)([conv4, upsa2])
    conv11 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merg2)
    batc11 = BatchNormalization(axis=-1)(conv11)
    acti11 = Activation('relu')(batc11)
    drop11 = Dropout(0.25)(acti11)
    conv12 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(drop11)
    batc12 = BatchNormalization(axis=-1)(conv12)
    acti12 = Activation('relu')(batc12)
    drop12 = Dropout(0.25)(acti12)

    upsa3 = UpSampling2D(2)(acti12)
    merg3 = Concatenate(axis=-1)([conv2, upsa3])
    conv13 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merg3)
    batc13 = BatchNormalization(axis=-1)(conv13)
    acti13 = Activation('relu')(batc13)
    drop13 = Dropout(0.25)(acti13)
    conv14 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(drop13)
    batc14 = BatchNormalization(axis=-1)(conv14)
    acti14 = Activation('relu')(batc14)
    
    convol = Conv2D(2, 1,padding='same', kernel_initializer='he_normal')(acti14)
    acti = Activation('softmax')(convol)

    model = Model(inputs=T1, outputs=acti)
   

    return model
