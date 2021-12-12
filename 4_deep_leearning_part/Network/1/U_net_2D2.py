from keras.models import Model
from keras.layers import Input, concatenate, Dropout, Reshape, Permute, Activation, ZeroPadding2D, Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.initializers import orthogonal, constant, he_normal
from keras.regularizers import l2


def unet(T1_shape):

    T1 = Input(shape=T1_shape)
    # encode
    # 256x256
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(T1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 128x128
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 64x64
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 32x32
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 16x16
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv5)
    

    # decode
    
    up7 = UpSampling2D(size=(2, 2))(conv5)
    up7 = concatenate([up7, conv4], axis=-1)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = concatenate([up8, conv3], axis=-1)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up9, conv2], axis=-1)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv9)

    up10 = UpSampling2D(size=(2, 2))(conv9)
    up10 = concatenate([up10, conv1], axis=-1)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up10)
    conv10 = Dropout(0.5)(conv10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv10)
    conv10 = Dropout(0.5)(conv10)

    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up10)
    conv10 = Dropout(0.2)(conv10)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv10)
    conv11 = Conv2D(2, (1, 1), padding='same')(conv10)
    conv11 = Activation('softmax')(conv11)

    model = Model(input=T1, output=conv11)

    return model
