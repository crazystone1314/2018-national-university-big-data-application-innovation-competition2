# -*- coding:utf-8 -*-

from data_handlers import *
from sklearn.externals import joblib

# 原始数据路径
testing_path = 'dataset/Semifinal-testing-final.csv'
# 读取原始数据
testing_data = pd.read_csv(testing_path, names=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5'])

#数据特征处理
testing = preprocess_features(testing_data)


#加载模型进行预测
mlp_model = joblib.load('mlp_model.pkl')
preds_class = mlp_model.predict(testing)


result = pd.DataFrame(preds_class)
# 将结果转化为整型
result_1 = result[0].apply(int)
result_2 = pd.DataFrame(result_1)
#将结果保存到文件当中
result_2.to_csv('dataset/dsjyycxds_preliminary.txt', index=False, header=False)