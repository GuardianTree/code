from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.models import Model
from keras.layers import *
from keras import initializers, layers
from .caps_layers import ConvCapsuleLayer, DeconvCapsuleLayer, Mask, Length1


def _squash(input_tensor):
    """
    Activation that squashes capsules to lengths between 0 and 1, where more significant capsules are closer to a lenght of 1 and less significant capsules are pushed towards 0
    * computes the raw norm; most likely need to change for a safer norm
    """
    norm = tf.norm(input_tensor, axis=-1, keepdims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))


def caps_layer(x,kernel_size,num_capsule, num_atoms, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal'):
    input_height = x[1]
    input_width = x[2]
    input_num_capsule = x[3]
    input_num_atoms = x[4]
    W = add_weight(shape=[kernel_size,kernel_size,input_num_atoms, num_capsule *num_atoms],
                                 initializer=kernel_initializer,name='W')
    b = add_weight(shape=[1, 1, num_capsule, num_atoms],initializer=initializers.constant(0.1),name='b')
    # changed to [input_n_capsules, None, h,w, input_n_atoms]
    input_transposed = tf.transpose(x, [3, 0, 1, 2, 4])
    input_shape = K.shape(input_transposed)
    # n_capsules multiplied with n_samples


def unet(T1_shape):
    f=32
    T1 = Input(shape=T1_shape)
    _, H, W, C = T1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(T1)

    '''downsample_T1'''
    # 576x576  kernel_size=5, num_capsule=2, num_atoms=16, strides=2, padding='same',routings=1, name='primarycaps'
    conv1=ConvCapsuleLayer(kernel_size=3,num_capsule=2, num_atoms=16, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(conv1_reshaped)
    print('conv1=',conv1) #
    #conv1=conv_block(T1,f)
    maxp1 = MaxPool3D((2,2,1))(conv1)    #1
    print('maxp1=',maxp1)
    # 288x288
    
    conv2 = ConvCapsuleLayer(kernel_size=3,num_capsule=4, num_atoms=16, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(maxp1)
    conv2 = ConvCapsuleLayer(kernel_size=3,num_capsule=4, num_atoms=32, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(conv2)
    maxp2 = MaxPool3D((2,2,1))(conv2)    #2

    # 144x144
    conv3 = ConvCapsuleLayer(kernel_size=3,num_capsule=4, num_atoms=32, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(maxp2)
    conv3 = ConvCapsuleLayer(kernel_size=3,num_capsule=4, num_atoms=64, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(conv3)
    maxp3 = MaxPool3D((2,2,1))(conv3)#3
    
    # 72x72
    conv4 = ConvCapsuleLayer(kernel_size=3,num_capsule=8, num_atoms=64, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(maxp3)
    conv4 = ConvCapsuleLayer(kernel_size=3,num_capsule=8, num_atoms=128, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(conv4)
    print('conv4=',conv4)
    maxp4 = MaxPool3D((2,2,1))(conv4)    #4

    # 36x36
    conv5 =ConvCapsuleLayer(kernel_size=3,num_capsule=8, num_atoms=128, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(maxp4)   #5

    '''upsample_T1'''

    upsa1 = UpSampling3D((2,2,1))(conv5) #4
    print('upsa1=',upsa1)
    merg1 = Concatenate(axis=-2)([conv4, upsa1])
    print('merg1=',merg1)
    conv6=ConvCapsuleLayer(kernel_size=3,num_capsule=8, num_atoms=128, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(merg1)
    conv6=ConvCapsuleLayer(kernel_size=3,num_capsule=8, num_atoms=64, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(conv6)

    upsa2 = UpSampling3D((2,2,1))(conv6) #3
    merg2 = Concatenate(axis=-2)([conv3, upsa2])
    conv7=ConvCapsuleLayer(kernel_size=3,num_capsule=4, num_atoms=64, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(merg2)
    conv7=ConvCapsuleLayer(kernel_size=3,num_capsule=4, num_atoms=32, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(conv7)

    upsa3 = UpSampling3D((2,2,1))(conv7) #2
    merg3 = Concatenate(axis=-2)([conv2, upsa3])
    conv8=ConvCapsuleLayer(kernel_size=3,num_capsule=4, num_atoms=32, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(merg3)
    conv8=ConvCapsuleLayer(kernel_size=3,num_capsule=4, num_atoms=16, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(conv8)
    
    upsa4 = UpSampling3D((2,2,1))(conv8) #1
    merg4 = Concatenate(axis=-2)([conv1, upsa4])
    conv9=ConvCapsuleLayer(kernel_size=3,num_capsule=2, num_atoms=16, strides=1, padding='same', 
               routings=3,kernel_initializer='he_normal')(merg4)
    print('conv9=',conv9)
    out_seg =(Length1(num_classes=2, seg=True, name='out_seg')(conv9))
    #convol = Conv2D(2, 1,padding='same')(conv9)
    #acti = Activation('softmax')(convol)
    model = Model(inputs=T1, outputs=out_seg)
   

    return model
