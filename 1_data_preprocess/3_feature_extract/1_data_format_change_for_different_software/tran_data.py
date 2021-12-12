import pandas as pd
import cv2
import openpyxl
import xlwt
import numpy as np
import numpy
import os
import matplotlib.pyplot as plt
from math import pow, floor
from time import *
import scipy.io as scio

def load_mat(path,i):
    train_mat = path+str(i)+'.mat'
    data = scio.loadmat(train_mat)
    img=data['data']
    return img

def extract_data(path_a,path_b,save_path,n,xline_l):
    y=165888
    z=16
    for i in range(n):
        Data = np.zeros(( y, z), dtype=np.float64)
        print(Data.shape)
        feature = load_mat(path_a,i) #(1,165888)
        #print('feature=',feature[0].shape)
        Index = load_mat(path_b,i)#(165888,17)
        #print('Index=',Index.shape)
        for j in range(y):
            #print("ok?")
            for k in range(z):
                #print('index=',Index[j][k])
                index=Index[j][k]
                Data[j][k]=feature[0][index]
                #print("Data[j][k]=",Data[j][k])
        #print(Data.shape)
        datapath=save_path+str(i)+'.mat'
        print(datapath)
        scio.savemat(datapath,{'data':Data})
if __name__ == '__main__':
     #####  data_load #####
     path1='/storage/student22/3hinge_seg/1000/test/test/sulc/rh/'
     path2='/storage/student22/3hinge_seg/1000/test/test/coor/rh/'
     tar_path='/storage/student22/3hinge_seg/1000/test/test/new/sulc/rh/'
     xlen =100
     xline_l = 165888
     begin_time = time()
     extract_data(path1,path2,tar_path,xlen,xline_l)
     end_time = time()
     run_time = end_time - begin_time
     print('the code cost time is :', run_time)
  
