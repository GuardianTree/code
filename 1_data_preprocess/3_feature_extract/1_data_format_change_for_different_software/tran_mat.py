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

def loaddata_m(x1, xlen, xline_l,path): #
    # data load
    df1 = pd.read_excel(x1)
    print("data1ok?")
    for i in range(xlen):
        len_l=len(df1["data" + str(i)].values)
        xx = np.zeros((len_l), dtype=np.float64)
        xx = df1["data" + str(i)].values
        datapath=path+str(i)+'.mat'
        print(datapath)
        scio.savemat(datapath,{'data':xx})




if __name__ == '__main__':

    filepath='/storage/student22/3hinge_seg/1000/test/test/volume/rh/'
    
    #####  data_load #####
    # T1
    train_A = '/storage/student22/3hinge_seg/1000/test/volume1.xlsx'
    
    xlen = 100
   
    xline_l = 165888
 
    loaddata_m(train_A, xlen, xline_l,filepath)
    
    

  
