from keras.models import Model
from keras.layers import *
from keras.initializers import orthogonal, constant, he_normal
from keras import regularizers


def con_block(data, f, k, dr, momen):
    x = Conv1D(f, k, padding='same', kernel_initializer='he_normal')(data)
    x = BatchNormalization(axis=-1, momentum=momen)(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)
    x = Conv1D(f, k, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1, momentum=momen)(x)
    x = Activation('relu')(x)
    return x


def unet(T1_shape, X_shape, Y_shape, Z_shape):
    f = 32
    k = 3
    dr = 0.3
    momen = 0.6

    T1 = Input(shape=T1_shape)
    coor_X = Input(shape=X_shape)
    coor_Y = Input(shape=Y_shape)
    coor_Z = Input(shape=Z_shape)
    '''coor+T1'''
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(T1)
    batc1 = BatchNormalization(axis=-1, momentum=0.6)(conv1)
    acti1 = Activation('relu')(batc1)

    conv1_x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(coor_X)
    batc1_x = BatchNormalization(axis=-1, momentum=0.6)(conv1_x)
    acti1_x = Activation('relu')(batc1_x)

    conv1_y = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(coor_Y)
    batc1_y = BatchNormalization(axis=-1, momentum=0.6)(conv1_y)
    acti1_y = Activation('relu')(batc1_y)

    conv1_z = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(coor_Z)
    batc1_z = BatchNormalization(axis=-1, momentum=0.6)(conv1_z)
    acti1_z = Activation('relu')(batc1_z)

    con = Concatenate(axis=-1)([acti1, acti1_x, acti1_y, acti1_z])



    '''downsample_T1'''
    # 1
    conv1 = con_block(con, f * 1, k, dr, momen)
    maxp1 = MaxPool1D(2)(conv1)  # 1 4096
    # 2
    conv2 = con_block(maxp1, f * 2, k, dr, momen)
    maxp2 = MaxPool1D(2)(conv2)  # 2 2048
    # 3
    conv3 = con_block(maxp2, f * 4, k, dr, momen)
    maxp3 = MaxPool1D(2)(conv3)  # 3 1024
    # 4
    conv4 = con_block(maxp3, f * 8, k, dr, momen)
    maxp4 = MaxPool1D(2)(conv4)  # 4 512
    # 5
    conv5 = con_block(maxp4, f * 16, k, dr, momen)
    maxp5 = MaxPool1D(2)(conv5)  # 5 256
    # 6
    conv6 = con_block(maxp5, f * 16, k, dr, momen)

    '''upsample_T1'''
    # 5'
    upsa_5 = UpSampling1D(2)(conv6)  # 512
    merg_5 = Concatenate(axis=-1)([conv5, upsa_5])
    conv_5 = con_block(merg_5, f * 16, k, dr, momen)

    # 4'
    upsa_4 = UpSampling1D(2)(conv_5)  # 1024
    merg_4 = Concatenate(axis=-1)([conv4, upsa_4])
    conv_4 = con_block(merg_4, f * 8, k, dr, momen)

    # 3'
    upsa_3 = UpSampling1D(2)(conv_4)  # 2048
    merg_3 = Concatenate(axis=-1)([conv3, upsa_3])
    conv_3 = con_block(merg_3, f * 4, k, dr, momen)

    # 2'
    upsa_2 = UpSampling1D(2)(conv_3)  # 4096
    merg_2 = Concatenate(axis=-1)([conv2, upsa_2])
    conv_2 = con_block(merg_2, f * 2, k, dr, momen)

    # 1'
    upsa_1 = UpSampling1D(2)(conv_2)  # 8192
    merg_1 = Concatenate(axis=-1)([conv1, upsa_1])
    conv_1 = con_block(merg_1, f * 1, k, dr, momen)

    convol = Conv1D(2, 1, padding='same')(conv_1)
    acti = Activation('softmax')(convol)

    model = Model(inputs=[T1, coor_X, coor_Y, coor_Z], outputs=acti)
    return model
