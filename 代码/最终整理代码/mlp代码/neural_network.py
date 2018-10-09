# -*- coding:utf-8 -*-

from data_handlers import *
from sklearn.neural_network import MLPClassifier

# 原始数据路径
training_path = 'dataset/training-final.csv'
source_data_path = 'dataset/Semifinal-testing-final.csv'

# 读取原始数据
# training-final数据
training_data = pd.read_csv(training_path, names=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand'])
# Semifinal-testing-final数据
testing_data = pd.read_csv(source_data_path, names=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5'])

#数据特征处理
training = preprocess_features(training_data)
#数据不均衡处理
X, y = imblanced_process(training)

params={
    'hidden_layer_zises': (40, 40, 40, 40),
    'solver': 'adam',
    'activation': 'tanh',
    'alpha': 0.0001,
    'learning_rate_init': 0.001,
    'max_iter': 400
}
mlp_model = MLPClassifier()
# 设置模型参数
mlp_model.set_params(**params)
# 训练模型
mlp_model.fit(X, y)
#用模型进行预测
preds_class = mlp_model.predict(testing_data)

result = pd.DataFrame(preds_class)
# 将结果转化为整型
result_1 = result[0].apply(int)
result_2 = pd.DataFrame(result_1)
# 将数据保存到文件dsjyycxds_preliminary.txt中
target_path = 'dataset/dsjyycxds_preliminary.txt'
result_2.to_csv(target_path, index=False, header=False)