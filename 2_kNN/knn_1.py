import pandas as pd
import numpy as np
import os
import xlwt
import openpyxl
from time import *
import math

from sklearn.neighbors import NearestNeighbors
import scipy.io as scio


def load_coor(path, n):
    hinge = pd.read_excel(path)
    print('coor load OK!!')
    lenver = len(hinge["ver" + str(0)].values)
    a = 0
    x1 = np.zeros((n, lenver), dtype=np.float64)
    b = 1
    y1 = np.zeros((n, lenver), dtype=np.float64)
    c = 2
    z1 = np.zeros((n, lenver), dtype=np.float64)
    for i in range(n):
        x1[i] = hinge["ver" + str(a)].values
        y1[i] = hinge["ver" + str(b)].values
        z1[i] = hinge["ver" + str(c)].values

        a = a + 3
        b = b + 3
        c = c + 3

    return (np.array(x1)), (np.array(y1)), (np.array(z1))


def get_orig(x, y, z, n):
    for a in range(n):
        lenx = len(x[a])

    A = np.zeros((n, lenx, 3), dtype=np.float64)

    for i in range(n):
        # print(len(x[i]))
        for j in range(len(x[i])):
            A[i][j][0] = (x[i][j])
            A[i][j][1] = (y[i][j])
            A[i][j][2] = (z[i][j])

    return A


def K_neig(A, n):
    index_all = []
    for i in range(n):
        print('i=', i)
        samples = A[i].tolist()
        neigh = NearestNeighbors(n_neighbors=17)
        neigh.fit(samples)
        index_1 = []
        for j in range(len(A[i])):  # len(A[i])
            my_coor = A[i][j]
            # print(my_coor)
            (dis, index) = neigh.kneighbors([my_coor])
            # print('dis=',dis[0])
            # print('index=',index[0])
            index_1.append(index[0])
        index_all.append(index_1)
    Index = np.array(index_all)
    # print(Index.shape)
    return Index


def save_m(data, n, path):  # Index=(1,165888,17)
    for i in range(n):
        Index = data[i]
        # print(Index.shape)
        datapath = path + str(i + 300) + '.mat'  # 300-499
        # print(datapath)
        scio.savemat(datapath, {'data': Index})


if __name__ == '__main__':
    # coor data load
    path_coor = '/storage/ZLL/code/label_test/coor/vector/coor/coor_21.xlsx'
    path_target = '/storage/ZLL/code/label_test/coor/vector/coor/rh/'

    n = 200
    (x0, y0, z0) = load_coor(path_coor, n)
    A = get_orig(x0, y0, z0, n)
    print(A.shape)
    begin_time = time()
    Index = K_neig(A, n)
    end_time = time()
    run_time = end_time - begin_time
    save_m(Index, n, path_target)
    print('the code cost time is :', run_time)
