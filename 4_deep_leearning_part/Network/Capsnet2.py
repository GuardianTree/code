from keras.models import Model
from keras.layers import *
from keras.initializers import orthogonal, constant, he_normal
from keras import regularizers
from keras import initializers, layers
from .caps_layers import ConvCapsuleLayer,DeconvCapsuleLayer, Mask, Length

def conv(x,f):
    conv_1 = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(x)
    batc_1 = BatchNormalization(axis=-1, momentum=0.6)(conv_1)
    acti_1 = Activation('relu')(batc_1)
    drop_1 = Dropout(0.2)(acti_1)
    conv_2 = Conv2D(f, 3, padding='same', kernel_initializer='he_normal')(drop_1)
    batc_2 = BatchNormalization(axis=-1, momentum=0.6)(conv_2)
    acti_2 = Activation('relu')(batc_2)
    return acti_2


def unet(T1_shape):
    f=32
    upsamp_type = 'deconv'
     
    T1 = Input(shape=T1_shape)
    _, H, W, C =T1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(T1)
    conv1_reshaped = ConvCapsuleLayer(kernel_size=3, num_capsule=2, num_atoms=16, strides=1, padding='same',
                                    routings=1, name='primarycaps0')(conv1_reshaped)
    
    #conv1_reshaped =(64,64,1,16)
    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=3, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)
    print('primary_caps=',primary_caps) #primary_caps=64, 64, 2, 16
    # Layer 2: Convolutional Capsule
    conv_cap_2_1 = ConvCapsuleLayer(kernel_size=3, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=1, name='conv_cap_2_1')(primary_caps)
    print('conv_cap_2_1=',conv_cap_2_1)#conv_cap_2_1=32, 32, 4, 16

    # Layer 2: Convolutional Capsule
    conv_cap_2_2 = ConvCapsuleLayer(kernel_size=3, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=1, name='conv_cap_2_2')(conv_cap_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap_3_1 = ConvCapsuleLayer(kernel_size=3, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=1, name='conv_cap_3_1')(conv_cap_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap_3_2 = ConvCapsuleLayer(kernel_size=3, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                    routings=1, name='conv_cap_3_2')(conv_cap_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap_4_1 = ConvCapsuleLayer(kernel_size=3, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=1, name='conv_cap_4_1')(conv_cap_3_2)
    #print('conv_cap_4_1 =',conv_cap_4_1 )
    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=3, num_capsule=8, num_atoms=32, upsamp_type=upsamp_type,
                                        scaling=2, padding='same', routings=1,
                                        name='deconv_cap_1_1')(conv_cap_4_1)

    # Skip connection
    up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=3, num_capsule=4, num_atoms=32, strides=1,
                                      padding='same', routings=1, name='deconv_cap_1_2')(up_1)

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=3, num_capsule=4, num_atoms=16, upsamp_type=upsamp_type,
                                        scaling=2, padding='same', routings=1,
                                        name='deconv_cap_2_1')(deconv_cap_1_2)

    # Skip connection
    up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=3, num_capsule=4, num_atoms=16, strides=1,
                                      padding='same', routings=1, name='deconv_cap_2_2')(up_2)

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=3, num_capsule=2, num_atoms=16, upsamp_type=upsamp_type,
                                        scaling=2, padding='same', routings=1,
                                        name='deconv_cap_3_1')(deconv_cap_2_2)

    # Skip connection
    up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1,  conv1_reshaped])

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=2, num_atoms=16, strides=1, padding='same',
                                routings=1, name='seg_caps')(up_3)

    '''downsample_T1'''
    # 576x576  kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',routings=1, name='primarycaps'
   # conv_1=ConvCapsuleLayer(kernel_size=3,num_capsule=2, num_atoms=32, strides=1, padding='same', 
               #routings=3,kernel_initializer='he_normal')(conv1_reshaped)
    
    out_seg =(Length(num_classes=2, seg=True, name='out_seg')(seg_caps))
    #print('T1.shape=',T1.shape)
    #print('out_seg=',out_seg.shape)
    model = Model(inputs=T1, outputs=out_seg)
   

    return model
