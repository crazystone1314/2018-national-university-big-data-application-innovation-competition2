# -*- coding:utf-8 -*-

from data_handlers import *
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# 原始数据路径
training_path = 'dataset/training-final.csv'

# 读取原始数据
training_data = pd.read_csv(training_path, names=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand'])


#数据特征处理
training = preprocess_features(training_data)
#数据不均衡处理
X_resampled, y_resampled = imblanced_process(training)
#将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=42)

# 设置模型参数
params={
    'hidden_layer_sizes': (40, 40, 40, 40),
    'solver': 'adam',
    'activation': 'tanh',
    'alpha': 0.0001,
    'learning_rate_init': 0.001,
    'max_iter': 300
}
mlp = MLPClassifier(random_state=35)
mlp.set_params(**params)

#使用交叉验证来获取平均准确率分数
scores = cross_val_score(mlp, X_resampled, y_resampled, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print("平均准确率为：{0:.8f}%".format(average_accuracy))


# 模型评估报告
print("\n模型评估报告如下：")
# 训练模型
mlp_model = mlp.fit(X_train, y_train)
# 模型在测试集上的预测
preds_class = mlp_model.predict(X_test)
preds_proba = mlp_model.predict_proba(X_test)
# 模型的预测准确率和损失率
print("Accuracy: {}".format(metrics.accuracy_score(y_test,preds_class)))
print("Logloss: {}".format(metrics.log_loss(y_test,preds_proba)))
print(metrics.confusion_matrix(y_test,preds_class))
print(metrics.classification_report(y_test, preds_class))


#将训练好的模型保存到文件当中
mlp_model = mlp.fit(X_resampled, y_resampled)
joblib.dump(mlp_model, "mlp_model.pkl")
