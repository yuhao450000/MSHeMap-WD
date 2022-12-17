'''
Author: your name
Date: 2021-06-30 00:50:58
LastEditTime: 2021-06-30 01:00:46
LastEditors: Please set LastEditors
Description: In User Settings Edit
'''
# -*- coding: utf-8 -*-
import numpy as np
from numpy import NaN, savetxt
import pandas as pd
import datetime
import random
# import SMOTE
import dpdata
# from sklearn import preprocessing
from dpdata import DataforClassify
from dpdata import MSHeMap
import distance
import matplotlib.pyplot as plt
distanceResList = []


def zscore(matrix):
    for i, value in enumerate(range(matrix.shape[1])):
        matrix[:, i] = (
            matrix[:, i] -
            np.average(matrix[:, i])) / np.std(
            matrix[:, i], ddof=1)
    return matrix


def cal_distance(target_data_index):
    res_frame = []
    # classify_result_list = []
    # data[0]做为T, 其他除本项目的数据集分别做为S,
    # target_data_index = 0
    for S_index, datai in enumerate(data):
        if True:  # 如果数据项目相同就不要算了
            # if datai.group == data[target_data_index].group:  # 如果数据项目相同就不要算了
            # continue
            # else:
            source_data_index = S_index
            filename = './res_step1/res_BT_BS/B_S' + \
                str(target_data_index) + ' by ' + str(source_data_index)
            # np.save(filename+'.npy', self.B_S)
            B_S = np.load(filename+'.npy')

            filename = './res_step1/res_BT_BS/B_T' + \
                str(target_data_index) + ' by ' + str(source_data_index)
            # np.save(filename+'.npy', self.B_T)
            B_T = np.load(filename+'.npy')

            # 此时的BT BS就读上来了，在进行距离计算
            #
            # CD=distance.cosine_distance(B_T,B_S)
            # ED=distance.EuclideanDistances(B_T,B_S)
            # C=distance.corre(B_T,B_S)
            # dis = distance.compute_distances_one_loop(B_T,B_S)
            dis = distance.SinkhornDistance_per_loop(B_T, B_S)
            dis_output_filename = './res_step1/res_BT_BS/dis' + \
                str(target_data_index) + ' by ' + str(source_data_index)+'.csv'
            disName = 'dis' + str(target_data_index) + \
                ' by ' + str(source_data_index)
            # 距离计算结果，输出到 文件名为 dis_output_filename的csv文件里，待补充-20210630yh
            # savetxt('dis.csv', dis, fmt='%f', delimiter=',')
            print(filename, dis)
            distanceResList.append(str(dis))


distanceSTResList = []


def cal_distance_between_ST(target_data_index):

    for S_index, datai in enumerate(data):
        # if True:  # 如果数据项目相同就不要算了
        if datai.group != data[target_data_index].group:  # 如果数据项目相同就不要算了
            distanceSTResList.append(str(NaN))
            # continue
        else:
            source_data_index = S_index
            filename = './res_step1/T_S_instance/S' + \
                str(target_data_index) + ' by ' + str(source_data_index)
            # np.save(filename+'.npy', self.B_S)
            S = np.load(filename+'.npy')

            filename = './res_step1/T_S_instance/T' + \
                str(target_data_index) + ' by ' + str(source_data_index)
            # np.save(filename+'.npy', self.B_T)
            T = np.load(filename+'.npy')

            # 此时的T S就读上来了，在进行距离计算
            #
            # CD=distance.cosine_distance(B_T,B_S)
            # ED=distance.EuclideanDistances(B_T,B_S)
            # C=distance.corre(B_T,B_S)
            # dis = distance.compute_distances_one_loop(B_T,B_S)
            T = zscore(T)
            S = zscore(S)
            dis = distance.SinkhornDistance_per_loop(T, S)
            dis_output_filename = './res_step1/res_T_S/dis' + \
                str(target_data_index) + ' by ' + str(source_data_index)+'.csv'
            disName = 'dis' + str(target_data_index) + \
                ' by ' + str(source_data_index)
            # 距离计算结果，输出到 文件名为 dis_output_filename的csv文件里，待补充-20210630yh
            # savetxt('dis.csv', dis, fmt='%f', delimiter=',')
            print(filename, dis)
            distanceSTResList.append(str(dis))
    return distanceSTResList


def step0():
    for T_index in range(len(data)):
        target_data_index = T_index
        cal_distance_between_ST(target_data_index)
        print("step0:their distance of S T:" + data[T_index].dataName +
              "done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    df = pd.DataFrame(np.array(distanceSTResList).reshape(28, 28))
    df.to_csv('./res_step1/T_S_instance/disRes.csv')
    # 明天接着弄这里，T,S已经存好


def step2():
    for T_index in range(len(data)):
        target_data_index = T_index
        cal_distance(target_data_index)
        print("step2:their distance" + data[T_index].dataName +
              "done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    df = pd.DataFrame(np.array(distanceResList).reshape(28, 28))
    df.to_csv('./res_step1/res_BT_BS/disRes.csv')


def step3():

    df = pd.read_csv('./res_step1/T_S_instance/disRes.csv', index_col=False)
    print(df.values[:, 1:])
    C = df.values[:, 1:]
    plt.imshow(C, origin='lower')

    plt.title('Sinkhorn Distance matrix')
    # axisTicks =
    # print(axisTicks)

    plt.colorbar()

    # ax = plt.gca()

    # label = np.arange(1, 29, 1)  # 填写自己的标签

    # ax.set_xticklabels(label)
    plt.show()


starttime = datetime.datetime.now()
data = np.load('data210629.npy', allow_pickle=True)

dpdata.disdata(data)
print('This is a new step2 of HeMap test')

step0()
# step 2: calculate BTn, BSn,their distance
# step2()
step3()
