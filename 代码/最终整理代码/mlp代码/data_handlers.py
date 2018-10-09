# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler


def transform_jkq(x):
    '''
    将J、Q、K映射成11、12、13
    :param x:
    :return:
    '''
    if x == 'J':
        return 11
    elif x == 'Q':
        return 12
    elif x == 'K':
        return 13
    else:
        return x


def bincount2D_vectorized(a):
    '''
    计算四种花色的数量和13种排名的有无
    :param a:
    :return:
    '''
    N = a.max() + 1
    a_offs = a + np.arange(a.shape[0])[:, None] * N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0] * N).reshape(-1, N)

def preprocess_features(input_data_df):
    #将J、Q、K映射成11、12、13
    input_data_df['C1'] = input_data_df['C1'].apply(transform_jkq)
    input_data_df['C2'] = input_data_df['C2'].apply(transform_jkq)
    input_data_df['C3'] = input_data_df['C3'].apply(transform_jkq)
    input_data_df['C4'] = input_data_df['C4'].apply(transform_jkq)
    input_data_df['C5'] = input_data_df['C5'].apply(transform_jkq)

    # 将C、D、H、S 映射为1、2、3、4
    encode_map = {'C': 1, 'D': 2, 'H': 3, 'S': 4}
    input_data_df['S1'] = input_data_df['S1'].map(encode_map)
    input_data_df['S2'] = input_data_df['S2'].map(encode_map)
    input_data_df['S3'] = input_data_df['S3'].map(encode_map)
    input_data_df['S4'] = input_data_df['S4'].map(encode_map)
    input_data_df['S5'] = input_data_df['S5'].map(encode_map)

    # 计算四种花色的数量
    S_training = input_data_df.iloc[:, [0, 2, 4, 6, 8]].astype(int)
    S_training = pd.DataFrame(bincount2D_vectorized(S_training.values), columns=['suitCount0', 'suitCount1', 'suitCount2', 'suitCount3', 'suitCount4'])
    input_data_df = pd.merge(input_data_df, S_training, how='left', left_index=True, right_index=True).drop(['suitCount0'], axis=1)
    # 计算13种排名的有无
    R_training = input_data_df.iloc[:, np.arange(1, 10, 2)].astype(int)
    cols = ['rank{}'.format(x) for x in range(0, 14, 1)]
    R_training = pd.DataFrame(bincount2D_vectorized(R_training.values), columns=cols)
    input_data_df = pd.merge(input_data_df, R_training, how='left', left_index=True, right_index=True).drop(['rank0'], axis=1)

    #各种排名的种类数
    R_training = input_data_df.loc[:, ['rank{}'.format(n) for n in range(1, 14, 1)]].astype(int)
    R_training = pd.DataFrame(bincount2D_vectorized(R_training.values), columns=['rankCount{}'.format(n) for n in range(0, 5, 1)])
    input_data_df = pd.merge(input_data_df, R_training, how='left', left_index=True, right_index=True).drop(['rankCount0'], axis=1)

    #13种排名各排名之间的差值的绝对值
    input_data_df['diff1_13'] = np.abs(input_data_df['rank1'] - input_data_df['rank13'])
    for i in range(2, 14, 1):
        input_data_df['diff{}_{}'.format(i, i - 1)] = np.abs(input_data_df['rank{}'.format(i)] - input_data_df['rank{}'.format(i - 1)])

    # 删除原始特征和13种花色的有无
    out_data_df = input_data_df.drop(['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5'], axis=1)
    #out_data_df = out_data_df.drop(['rank{}'.format(n) for n in range(1, 14, 1)], axis=1)

    return out_data_df






def imblanced_process(dataset):
    """
    不平衡数据处理
    :param dataset:
    :return:
    """
    X = dataset.drop(['hand'], axis=1)
    y = dataset.hand
    ros = RandomOverSampler(random_state=14)
    X_resampled, y_resampled = ros.fit_sample(X, y)
    return X_resampled, y_resampled
