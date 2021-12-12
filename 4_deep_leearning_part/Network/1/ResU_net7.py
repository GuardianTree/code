
from keras.models import Model

from keras.optimizers import Adam
from keras import backend as K
from keras.layers import *

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True, name=None):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x
    

def bottleneck_Block(input, nb_filters, strides=(1, 1), with_conv_shortcut=False, use_activation=True):
    k1, k2, k3 = nb_filters
    x = Conv2d_BN(input, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same', use_activation=False)

    if with_conv_shortcut:
        shortcut = Conv2d_BN(input, nb_filter=k3, kernel_size=1, strides=strides, use_activation=False)
        x = add([x, shortcut])
    else:
        x = add([x, input])

    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x
 

def DAC(input):
    init = input
    bound_1 = AtrousConvolution2D(2048, 3, 3, atrous_rate=(1, 1), activation='relu')(init)
    bound_1 =ZeroPadding2D(padding=(1,1),dim_ordering='default')(bound_1)

    bound_2 = AtrousConvolution2D(2048, 3, 3, atrous_rate=(3, 3), activation='relu')(init)
    bound_2 = AtrousConvolution2D(2048, 3, 3, atrous_rate=(1, 1), activation='relu')(bound_2)
    bound_2 = ZeroPadding2D(padding=(4,4),dim_ordering='default')(bound_2)

    bound_3 = AtrousConvolution2D(2048, 3, 3, atrous_rate=(1, 1), activation='relu')(init)
    bound_3 = AtrousConvolution2D(2048, 3, 3, atrous_rate=(3, 3), activation='relu')(bound_3)
    bound_3 = AtrousConvolution2D(2048, 3, 3, atrous_rate=(1,1), activation='relu')(bound_3)
    bound_3 =ZeroPadding2D(padding=(5,5),dim_ordering='default')(bound_3)

   
    #x = Merge([init,bound_1,bound_2,bound_3,bound_4], axis=-1,mode='concat')
    #x = concatenate([init,bound_1,bound_2,bound_3,bound_4], axis=-1)
    x=Add()([init, bound_1,bound_2 , bound_3])
    print(x.shape)
    return x


def RMP(input):
    init = input
    #ok
    p_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding="same")(init)
    p_1 = Conv2D(1,(1,1),padding="same",use_bias=False)(p_1)
    print(p_1.shape)
    p_1 = UpSampling2D(size=(2, 2),interpolation='bilinear')(p_1)
    #print(p_1.shape)
    
    p_2 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4),padding="same")(init)
    p_2 = Conv2D(1,(1,1),padding="same",use_bias=False)(p_2)
    print(p_2.shape)
    p_2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(p_2)
    #print(p_2.shape)
    
    #ok
    p_3 = MaxPooling2D(pool_size=(5, 5), strides=(5, 5),padding="same")(init)
    p_3 = Conv2D(1,(1,1),padding="same",use_bias=False)(p_3)
    print(p_3.shape)
    p_3 = UpSampling2D(size=(4, 4), data_format=None, interpolation='bilinear')(p_3)
    #print(p_3.shape)
    
    p_4 = MaxPooling2D(pool_size=(8, 8), strides=(8, 8),padding="same")(init)
    p_4 = Conv2D(1,(1,1),padding="same",use_bias=False)(p_4)
    print(p_4.shape)
    p_4 = UpSampling2D(size=(8, 8), data_format=None, interpolation='bilinear')(p_4)
    
    x = concatenate([init, p_1, p_2, p_3, p_4],axis=-1)
    print(x.shape)
    return x
    
def resnet_unet(height=576, width=576, channel=1, encoder='resnet50', dropout=True, classes=2):
    inputs = Input(shape=(height, width, channel))

    with K.name_scope('Encode_1'):
        conv1_1 = Conv2d_BN(inputs, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
        conv1_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_1)
    
    with K.name_scope('Encode_2'):
        conv2_1 = bottleneck_Block(conv1_2, nb_filters=[64, 64, 256], strides=(1, 1), with_conv_shortcut=True)
        conv2_2 = bottleneck_Block(conv2_1, nb_filters=[64, 64, 256])
        conv2_3 = bottleneck_Block(conv2_2, nb_filters=[64, 64, 256])
        
    with K.name_scope('Encode_3'):
        conv3_1 = bottleneck_Block(conv2_3, nb_filters=[128, 128, 512], strides=(2, 2), with_conv_shortcut=True)
        conv3_2 = bottleneck_Block(conv3_1, nb_filters=[128, 128, 512])
        conv3_3 = bottleneck_Block(conv3_2, nb_filters=[128, 128, 512])
        conv3_4 = bottleneck_Block(conv3_3, nb_filters=[128, 128, 512])
        
    with K.name_scope('Encode_4'):
        if encoder == 'resnet50':
            conv4_1 = bottleneck_Block(conv3_4, nb_filters=[256, 256, 1024], strides=(2, 2), with_conv_shortcut=True)
            conv4_2 = bottleneck_Block(conv4_1, nb_filters=[256, 256, 1024])
            conv4_3 = bottleneck_Block(conv4_2, nb_filters=[256, 256, 1024])
            conv4_4 = bottleneck_Block(conv4_3, nb_filters=[256, 256, 1024])
            conv4_5 = bottleneck_Block(conv4_4, nb_filters=[256, 256, 1024])
            conv4_6 = bottleneck_Block(conv4_5, nb_filters=[256, 256, 1024])
        elif encoder == 'resnet101':
            conv4_1 = bottleneck_Block(conv3_4, nb_filters=[256, 256, 1024], strides=(2, 2), with_conv_shortcut=True)
            conv4_2 = bottleneck_Block(conv4_1, nb_filters=[256, 256, 1024])
            conv4_3 = bottleneck_Block(conv4_2, nb_filters=[256, 256, 1024])
            conv4_4 = bottleneck_Block(conv4_3, nb_filters=[256, 256, 1024])
            conv4_5 = bottleneck_Block(conv4_4, nb_filters=[256, 256, 1024])
            conv4_6 = bottleneck_Block(conv4_5, nb_filters=[256, 256, 1024])
            conv4_7 = bottleneck_Block(conv4_6, nb_filters=[256, 256, 1024])
            conv4_8 = bottleneck_Block(conv4_7, nb_filters=[256, 256, 1024])
            conv4_9 = bottleneck_Block(conv4_8, nb_filters=[256, 256, 1024])
            conv4_10 = bottleneck_Block(conv4_9, nb_filters=[256, 256, 1024])
            conv4_11 = bottleneck_Block(conv4_10, nb_filters=[256, 256, 1024])
            conv4_12 = bottleneck_Block(conv4_11, nb_filters=[256, 256, 1024])
            conv4_13 = bottleneck_Block(conv4_12, nb_filters=[256, 256, 1024])
            conv4_14 = bottleneck_Block(conv4_13, nb_filters=[256, 256, 1024])
            conv4_15 = bottleneck_Block(conv4_14, nb_filters=[256, 256, 1024])
            conv4_16 = bottleneck_Block(conv4_15, nb_filters=[256, 256, 1024])
            conv4_17 = bottleneck_Block(conv4_16, nb_filters=[256, 256, 1024])
            conv4_18 = bottleneck_Block(conv4_17, nb_filters=[256, 256, 1024])
            conv4_19 = bottleneck_Block(conv4_18, nb_filters=[256, 256, 1024])
            conv4_20 = bottleneck_Block(conv4_19, nb_filters=[256, 256, 1024])
            conv4_21 = bottleneck_Block(conv4_20, nb_filters=[256, 256, 1024])
            conv4_22 = bottleneck_Block(conv4_21, nb_filters=[256, 256, 1024])
            conv4_23 = bottleneck_Block(conv4_22, nb_filters=[256, 256, 1024])
            
    with K.name_scope('Encode_5'):
        if encoder == 'resnet50':
            conv5_1 = bottleneck_Block(conv4_6, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
        elif encoder == 'resnet101':
            conv5_1 = bottleneck_Block(conv4_23, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
        conv5_2 = bottleneck_Block(conv5_1, nb_filters=[512, 512, 2048])
        conv5_3 = bottleneck_Block(conv5_2, nb_filters=[512, 512, 2048])
    


    with K.name_scope('DAC'):
        dac = DAC(conv5_3)
    
    with K.name_scope('RMP'):
        rmp = RMP(dac)

    with K.name_scope('Decode_1'):
        up6 = Conv2d_BN(UpSampling2D(size=(2, 2))(rmp), 1024, 2)
        if encoder == 'resnet50':
            merge6 = concatenate([conv4_6, up6], axis=-1)
        elif encoder == 'resnet101':
            merge6 = concatenate([conv4_23, up6], axis=-1)
        if dropout:
            merge6 = Dropout(0.5)(merge6)
        conv6 = Conv2d_BN(merge6, 512, 3)
        conv6 = Conv2d_BN(conv6, 512, 3)
    
    with K.name_scope('Decode_2'):
        up7 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv6), 512, 2)
        merge7 = concatenate([conv3_4, up7], axis=-1)
        if dropout:
            merge7 = Dropout(0.5)(merge7)
        conv7 = Conv2d_BN(merge7, 256, 3)
        conv7 = Conv2d_BN(conv7, 256, 3)
    
    with K.name_scope('Decode_3'):
        up8 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv7), 256, 2)
        merge8 = concatenate([conv2_3, up8], axis=-1)
        if dropout:
            merge8 = Dropout(0.5)(merge8)
        conv8 = Conv2d_BN(merge8, 128, 3)
        conv8 = Conv2d_BN(conv8, 128, 3)
        
    with K.name_scope('Decode_4'):
        up9 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv8), 64, 2)
        merge9 = concatenate([conv1_1, up9], axis=-1)
        if dropout:
            merge9 = Dropout(0.5)(merge9)
        conv9 = Conv2d_BN(merge9, 64, 3)
        conv9 = Conv2d_BN(conv9, 64, 3)
    conv10=UpSampling2D(size=(2, 2))(conv9)  
    conv10 = Conv2d_BN(conv10, 2, 1, use_activation=None)
    activation = Activation('softmax')(conv10)
    model = Model(inputs=inputs, outputs=activation)
    
    return model
