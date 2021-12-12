import keras.models as KM
import keras.layers as KL
import keras.engine as KE
import keras.backend as KB
from keras.utils import plot_model
from keras.preprocessing.image import img_to_array




import math
import numpy as np 

def branch(dilation_rate):
    x=KL.BatchNormalization()(input)
    x=KL.Activation('relu')(x)
    x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate,padding='same',
                kernel_initializer='he_normal')(x)
    x=KL.BatchNormalization()(x)
    x=KL.Activation('relu')(x)
    x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate, padding='same',
                kernel_initializer='he_normal')(x)
    return x
    
def ResBlock(input,filter,kernel_size,dilation_rates,stride):
    def branch(dilation_rate):
        x=KL.BatchNormalization()(input)
        x=KL.Activation('relu')(x)
        x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate, padding='same',
                    kernel_initializer='he_normal')(x)
        x=KL.BatchNormalization()(x)
        x=KL.Activation('relu')(x)
        x=KL.Conv2D(filter,kernel_size,strides=stride,dilation_rate=dilation_rate, padding='same',
                    kernel_initializer='he_normal')(x)
        return x
    out=[]
    for d in dilation_rates:
        out.append(branch(d))
    if len(dilation_rates)>1:
        out=KL.Add()(out)
    else:
        out=out[0]
    return out
def PSPPooling(input,filter):
    x1=KL.MaxPooling2D(pool_size=(2,2))(input)
    x2=KL.MaxPooling2D(pool_size=(4,4))(input)
    x3=KL.MaxPooling2D(pool_size=(8,8))(input)
    x4=KL.MaxPooling2D(pool_size=(16,16))(input)
    x1=KL.Conv2D(int(filter/4),(3,3), padding='same', kernel_initializer='he_normal')(x1)
    x2=KL.Conv2D(int(filter/4),(3,3), padding='same', kernel_initializer='he_normal')(x2)
    x3=KL.Conv2D(int(filter/4),(3,3), padding='same', kernel_initializer='he_normal')(x3)
    x4=KL.Conv2D(int(filter/4),(3,3), padding='same', kernel_initializer='he_normal')(x4)
    x1=KL.UpSampling2D(size=(2,2))(x1)
    x2=KL.UpSampling2D(size=(4,4))(x2)
    x3=KL.UpSampling2D(size=(8,8))(x3)
    x4=KL.UpSampling2D(size=(16,16))(x4)
    x=KL.Concatenate()([x1,x2,x3,x4,input])
    x=KL.Conv2D(filter,(3,3), padding='same', kernel_initializer='he_normal')(x)
    return x

def combine(input1,input2,filter):
    x=KL.Activation('relu')(input1)
    x=KL.Concatenate()([x,input2])
    x=KL.Conv2D(filter,(3,3), padding='same', kernel_initializer='he_normal')(x)
    return x

def resUnet(input_size = (576,576,1)):
    inputs=KM.Input(input_size)
    c1=x=KL.Conv2D(32,(3,3),strides=(1,1), padding='same', kernel_initializer='he_normal')(inputs)
    c2=x=ResBlock(x,32,(3,3),[1,3,15,31],(1,1))
    x=KL.Conv2D(64,(1,1),strides=(2,2))(x)
    c3=x=ResBlock(x,64,(3,3),[1,3,15,31],(1,1))
    x=KL.Conv2D(128,(1,1),strides=(2,2))(x)
    c4=x=ResBlock(x,128,(3,3),[1,3,15],(1,1))
    x=KL.Conv2D(256,(1,1),strides=(2,2))(x)
    c5=x=ResBlock(x,256,(3,3),[1,3,15],(1,1))
    x=KL.Conv2D(512,(1,1),strides=(2,2))(x)
    c6=x=ResBlock(x,512,(3,3),[1],(1,1))
    x=KL.Conv2D(1024,(1,1),strides=(2,2))(x)
    x=ResBlock(x,1024,(3,3),[1],(1,1))
    #x=PSPPooling(x,1024)
    x=KL.Conv2D(512,(3,3), padding='same', kernel_initializer='he_normal')(x)
    x=KL.UpSampling2D()(x)
    x=combine(x,c6,512)
    x=ResBlock(x,512,(3,3),[1],1)
    x=KL.Conv2D(256,(3,3), padding='same', kernel_initializer='he_normal')(x)
    x=KL.UpSampling2D()(x)
    x=combine(x,c5,256)
    x=ResBlock(x,256,(3,3),[1,3,15],1)
    x=KL.Conv2D(128,(3,3), padding='same', kernel_initializer='he_normal')(x)
    x=KL.UpSampling2D()(x)
    x=combine(x,c4,128)
    x=ResBlock(x,128,(3,3),[1,3,15],1)
    x=KL.Conv2D(64,(3,3), padding='same', kernel_initializer='he_normal')(x)
    x=KL.UpSampling2D()(x)
    x=combine(x,c3,64)
    x=ResBlock(x,64,(3,3),[1,3,15,31],1)
    x=KL.Conv2D(32,(3,3), padding='same', kernel_initializer='he_normal')(x)
    x=KL.UpSampling2D()(x)
    x=combine(x,c2,32)
    x=ResBlock(x,32,(3,3),[1,3,15,31],1)
    x=combine(x,c1,32)
    #x=PSPPooling(x,32)
    x=KL.Conv2D(2,(1,1),padding='same')(x)
    xc=KL.Activation('softmax')(x)
    model=KM.Model(inputs=inputs,outputs=xc)
    
    return model

