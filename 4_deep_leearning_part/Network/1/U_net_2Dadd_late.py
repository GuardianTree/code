from keras.layers import Input, BatchNormalization, MaxPool2D, Conv2D, UpSampling2D, Concatenate, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import *

def unet(T1_shape,T2_shape):

    T1 = Input(shape=T1_shape)
    T2 = Input(shape=T2_shape)
    #T3 = Concatenate(axis=-1)([T1, T2])

    '''downsample_T1'''
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(T1)
    batc1 = BatchNormalization(axis=-1)(conv1)
    acti1 = Activation('relu')(batc1)
    drop1 = Dropout(0.25)(acti1)
    conv2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(drop1)
    batc2 = BatchNormalization(axis=-1)(conv2)
    acti2 = Activation('relu')(batc2)
    drop2 = Dropout(0.25)(acti2)
    maxp1 = MaxPool2D(2)(drop2)


    conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(maxp1)
    batc3 = BatchNormalization(axis=-1)(conv3)
    acti3 = Activation('relu')(batc3)
    drop3 = Dropout(0.25)(acti3)
    conv4 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(drop3)
    batc4 = BatchNormalization(axis=-1)(conv4)
    acti4 = Activation('relu')(batc4)
    drop4 = Dropout(0.25)(acti4)
    maxp2 = MaxPool2D(2)(drop4)


    conv5 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(maxp2)
    batc5 = BatchNormalization(axis=-1)(conv5)
    acti5 = Activation('relu')(batc5)
    drop5 = Dropout(0.25)(acti5)
    conv6 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(drop5)
    batc6 = BatchNormalization(axis=-1)(conv6)
    acti6 = Activation('relu')(batc6)
    drop6 = Dropout(0.25)(acti6)
    maxp3 = MaxPool2D(2)(drop6)

    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(maxp3)
    batc7 = BatchNormalization(axis=-1)(conv7)
    acti7 = Activation('relu')(batc7)
    drop7 = Dropout(0.25)(acti7)
    conv8 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(drop7)
    batc8 = BatchNormalization(axis=-1)(conv8)
    acti8 = Activation('relu')(batc8)
    
    
    '''upsample'''
    upsa1 = UpSampling2D(2)(acti8)
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

    upsa2 = UpSampling2D(2)(drop10)
    merg2 = Concatenate(axis=-1)([conv4, upsa2])
    conv11 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merg2)
    batc11 = BatchNormalization(axis=-1)(conv11)
    acti11 = Activation('relu')(batc11)
    drop11 = Dropout(0.25)(acti11)
    conv12 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(drop11)
    batc12 = BatchNormalization(axis=-1)(conv12)
    acti12 = Activation('relu')(batc12)
    drop12 = Dropout(0.25)(acti12)

    upsa3 = UpSampling2D(2)(drop12)
    merg3 = Concatenate(axis=-1)([conv2, upsa3])
    conv13 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merg3)
    batc13 = BatchNormalization(axis=-1)(conv13)
    acti13 = Activation('relu')(batc13)
    drop13 = Dropout(0.25)(acti13)
    conv14 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(drop13)
    batc14 = BatchNormalization(axis=-1)(conv14)
    acti14 = Activation('relu')(batc14)



    '''downsample_T2'''
    conv1_T2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(T2)
    batc1_T2 = BatchNormalization(axis=-1)(conv1_T2)
    acti1_T2 = Activation('relu')(batc1_T2)
    drop1_T2 = Dropout(0.25)(acti1_T2)
    conv2_T2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(drop1_T2)
    batc2_T2 = BatchNormalization(axis=-1)(conv2_T2)
    acti2_T2 = Activation('relu')(batc2_T2)
    drop2_T2 = Dropout(0.25)(acti2_T2)
    maxp1_T2 = MaxPool2D(2)(drop2_T2)


    conv3_T2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(maxp1_T2)
    batc3_T2 = BatchNormalization(axis=-1)(conv3_T2)
    acti3_T2 = Activation('relu')(batc3_T2)
    drop3_T2 = Dropout(0.25)(acti3_T2)
    conv4_T2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(drop3_T2)
    batc4_T2 = BatchNormalization(axis=-1)(conv4_T2)
    acti4_T2 = Activation('relu')(batc4_T2)
    drop4_T2 = Dropout(0.25)(acti4_T2)
    maxp2_T2 = MaxPool2D(2)(drop4_T2)


    conv5_T2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(maxp2_T2)
    batc5_T2 = BatchNormalization(axis=-1)(conv5_T2)
    acti5_T2 = Activation('relu')(batc5_T2)
    drop5_T2 = Dropout(0.25)(acti5_T2)
    conv6_T2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(drop5_T2)
    batc6_T2 = BatchNormalization(axis=-1)(conv6_T2)
    acti6_T2 = Activation('relu')(batc6_T2)
    drop6_T2 = Dropout(0.25)(acti6_T2)
    maxp3_T2 = MaxPool2D(2)(drop6_T2)

    conv7_T2 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(maxp3_T2)
    batc7_T2 = BatchNormalization(axis=-1)(conv7_T2)
    acti7_T2 = Activation('relu')(batc7_T2)
    drop7_T2 = Dropout(0.25)(acti7_T2)
    conv8_T2 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(drop7_T2)
    batc8_T2 = BatchNormalization(axis=-1)(conv8_T2)
    acti8_T2 = Activation('relu')(batc8_T2)
    
    
    '''upsample'''
    upsa1_T2 = UpSampling2D(2)(acti8_T2)
    # print('upsam1 shape: ', upsam1.shape)
    merg1_T2 = Concatenate(axis=-1)([conv6_T2, upsa1_T2])
    conv9_T2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merg1_T2)
    batc9_T2 = BatchNormalization(axis=-1)(conv9_T2)
    acti9_T2 = Activation('relu')(batc9_T2)
    drop9_T2 = Dropout(0.25)(acti9_T2)
    conv10_T2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(drop9_T2)
    batc10_T2 = BatchNormalization(axis=-1)(conv10_T2)
    acti10_T2 = Activation('relu')(batc10_T2)
    drop10_T2 = Dropout(0.25)(acti10_T2)

    upsa2_T2 = UpSampling2D(2)(drop10_T2)
    merg2_T2 = Concatenate(axis=-1)([conv4_T2, upsa2_T2])
    conv11_T2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merg2_T2)
    batc11_T2 = BatchNormalization(axis=-1)(conv11_T2)
    acti11_T2 = Activation('relu')(batc11_T2)
    drop11_T2 = Dropout(0.25)(acti11_T2)
    conv12_T2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(drop11_T2)
    batc12_T2 = BatchNormalization(axis=-1)(conv12_T2)
    acti12_T2 = Activation('relu')(batc12_T2)
    drop12_T2 = Dropout(0.25)(acti12_T2)

    upsa3_T2 = UpSampling2D(2)(drop12_T2)
    merg3_T2 = Concatenate(axis=-1)([conv2_T2, upsa3_T2])
    conv13_T2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merg3_T2)
    batc13_T2 = BatchNormalization(axis=-1)(conv13_T2)
    acti13_T2 = Activation('relu')(batc13_T2)
    drop13_T2 = Dropout(0.25)(acti13_T2)
    conv14_T2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(drop13_T2)
    batc14_T2 = BatchNormalization(axis=-1)(conv14_T2)
    acti14_T2 = Activation('relu')(batc14_T2)

    T3 = Concatenate(axis=-1)([acti14,acti14_T2])
    convol = Conv2D(2, 1,padding='same', kernel_initializer='he_normal')(T3)
    acti = Activation('softmax')(convol)

    model = Model(inputs=[T1,T2], outputs=acti)
   

    return model
