from functools import reduce
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

batch_size=1

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


class Indices_Maxpool(keras.layers.Layer):
    def __init__(self, name):
        super(Indices_Maxpool, self).__init__(name=name)

    def call(self, inputs, **kwargs):

        val, index = inputs
        input_size = index.shape
        output_size = [x * 2 if i == 0 or i == 1 else x for i, x in enumerate(input_size[1:])]
        output = tf.reshape(tf.scatter_nd(tf.reshape(index, (-1, 1)), tf.reshape(val, (-1,)), (1 * np.prod(output_size),)), [-1] + output_size)

        return output


class Convs(keras.layers.Layer):
    def __init__(self, filters, name):
        super(Convs, self).__init__(name=name)
        self.blocks = keras.Sequential()
        self.blocks.add(keras.layers.Conv2D(filters, (3, 3), (1, 1), padding='same'))
        self.blocks.add(keras.layers.BatchNormalization())
        self.blocks.add(keras.layers.ReLU())

    def call(self, inputs, **kwargs):
        output = self.blocks(inputs)

        return output


def segnet(input_shape):
    #batch_size=1

    input_tensor = tf.keras.layers.Input(shape=input_shape, batch_size=1, name='input')
    x = input_tensor

    x1 = compose(Convs(64, name='encoder_conv1_1'),
                 Convs(64, name='encoder_conv1_2'))(x)
    val_1, index_1 = tf.nn.max_pool_with_argmax(x1, (1,2, 2,1), (1,2, 2,1), 'VALID', name='maxpool1')
    x2 = compose(Convs(128, name='encoder_conv2_1'),
                 Convs(128, name='encoder_conv2_2'))(val_1)
    val_2, index_2 = tf.nn.max_pool_with_argmax(x2, (1,2, 2,1), (1,2, 2,1), 'VALID', name='maxpool2')
    x3 = compose(Convs(256, name='encoder_conv3_1'),
                 Convs(256, name='encoder_conv3_2'),
                 Convs(256, name='encoder_conv3_3'))(val_2)
    val_3, index_3 = tf.nn.max_pool_with_argmax(x3, (1,2, 2,1),(1,2, 2,1), 'VALID', name='maxpool3')
    x4 = compose(Convs(512, name='encoder_conv4_1'),
                 Convs(512, name='encoder_conv4_2'),
                 Convs(512, name='encoder_conv4_3'))(val_3)
    val_4, index_4 = tf.nn.max_pool_with_argmax(x4,(1,2, 2,1),(1,2, 2,1), 'VALID', name='maxpool4')
    x5 = compose(Convs(512, name='encoder_conv5_1'),
                 Convs(512, name='encoder_conv5_2'),
                 Convs(512, name='encoder_conv5_3'))(val_4)
    val_5, index_5 = tf.nn.max_pool_with_argmax(x5, (1,2, 2,1),(1,2, 2,1), 'VALID', name='maxpool5')

    indices_maxpool5 = Indices_Maxpool(name='indices_maxpool5')([val_5, index_5])
    y5 = compose(Convs(512, name='decoder_conv5_1'),
                 Convs(512, name='decoder_conv5_2'),
                 Convs(512, name='decoder_conv5_3'))(indices_maxpool5)
    indices_maxpool4 = Indices_Maxpool(name='indices_maxpool4')([y5, index_4])
    y4 = compose(Convs(512, name='decoder_conv4_1'),
                 Convs(512, name='decoder_conv4_2'),
                 Convs(256, name='decoder_conv4_3'))(indices_maxpool4)
    indices_maxpool3 = Indices_Maxpool(name='indices_maxpool3')([y4, index_3])
    y3 = compose(Convs(256, name='decoder_conv3_1'),
                 Convs(256, name='decoder_conv3_2'),
                 Convs(128, name='decoder_conv3_3'))(indices_maxpool3)
    indices_maxpool2 = Indices_Maxpool(name='indices_maxpool2')([y3, index_2])
    y2 = compose(Convs(128, name='decoder_conv2_1'),
                 Convs(64, name='decoder_conv2_2'))(indices_maxpool2)
    indices_maxpool1 = Indices_Maxpool(name='indices_maxpool1')([y2, index_1])
    y1 = compose(Convs(64, name='decoder_conv1_1'),
                 Convs(21, name='decoder_conv1_2'))(indices_maxpool1)
    y2 = compose(Convs(2, name='decoder_conv_1'))(indices_maxpool1)
    print(y1)
    print(y2)
    #ad = tf.nn.conv2d(filter=[3,3,2,1],strides=[1,1,1,1], padding='SAME')(y1)
    output =tf.nn.softmax(y2)
    print(input_tensor)
    print(output)
    model  = tf.keras.Model(input_tensor, output, name='SegNet')

    return model

