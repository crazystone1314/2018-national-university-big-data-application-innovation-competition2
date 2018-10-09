# -*- coding:utf-8 -*-

from data_handlers import *
from feature_selection import create_traindata_and_testdata
from model_fusion import train_and_test_predictions
import pandas as pd
from xgboost import XGBClassifier

# 原始数据路径
training_path = 'dataset/training-final.csv'
# 读取原始数据
training_data = pd.read_csv(training_path, names=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand'])

# 原始数据路径
testing_path = 'dataset/Semifinal-testing-final.csv'
# 读取原始数据
testing_data = pd.read_csv(testing_path, names=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5'])

#数据特征处理
training = preprocess_features(training_data)
#数据特征处理
testing = preprocess_features(testing_data)

#数据不均衡处理
X_resampled, Y_resampled = imblanced_process(training)

X_resampled_df = pd.DataFrame(X_resampled.astype(int), columns=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5'])
Y_resampled_df = pd.DataFrame(Y_resampled.astype(int), columns=['hand'])
X_resampled_df['hand'] = Y_resampled_df
train_data = X_resampled_df

test_data = testing

#利用求得的重要特征从新划分训练集和测试集
train_data_X, train_data_Y, test_data_X = create_traindata_and_testdata(train_data, test_data)

#组合基模型预测的结果，构建训练集和测试集。
x_train, y_train, x_test = train_and_test_predictions(train_data_X, train_data_Y, test_data_X)

gbm = XGBClassifier(objective='multi:softmax ', num_class='multi:softprob')
gbm_model = gbm.fit(x_train,y_train)
preds_class = gbm_model.predict(x_test)
result = pd.DataFrame(preds_class)
# 将结果转化为整型
result_1 = result[0].apply(int)
result_2 = pd.DataFrame(result_1)
#将结果保存到文件当中
result_2.to_csv('dataset/dsjyycxds_preliminary.txt', index=False, header=False)