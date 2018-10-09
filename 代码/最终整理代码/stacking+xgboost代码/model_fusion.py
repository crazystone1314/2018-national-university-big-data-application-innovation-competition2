# -*- coding:utf-8 -*-


from sklearn.model_selection import KFold
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


def get_out_fold(clf, x_train, y_train, x_test):
    '''
    需要对每个基学习器使用K-fold，将Kge模型对Valid Set的预测结果拼起来，
    作为下一层学习器的输入。所以这里我们建立输出fold预测方法。
    :param clf:
    :param x_train:
    :param y_train:
    :param x_test:
    :return:
    '''
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    SEED = 23  # for reproducibility
    NFOLDS = 4  # set folds for out-of-fold prediction
    kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=False)

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def train_and_test_predictions(x_train, y_train, x_test):

    #构建不同的基学习器
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 15), solver='adam', activation='tanh', alpha=0.0001,
                        learning_rate_init=0.001, max_iter=400)
    rf = RandomForestClassifier(max_depth=32, min_samples_split=3, min_samples_leaf=2, verbose=0)
    et = ExtraTreesClassifier(n_estimators=150, max_depth=15, min_samples_leaf=2, verbose=0)
    # dt = DecisionTreeClassifier(max_depth=17)
    svm = SVC(kernel='rbf', gamma=0.1, C=10)

    #创建我们的OOF训练和测试预测。这些基础结果将被用作新的特征
    mlp_oof_train, mlp_oof_test = get_out_fold(mlp, x_train, y_train, x_test)  # MLP
    rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test)  # Random Forest
    et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test)  # Extra Trees
    # dt_oof_train,dt_oof_test = get_out_fold(dt,x_train,y_train,x_test)  #Decision Tree
    svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test)  # Support Vector

    #将各个基模型的预测结果作为特征合并起来。
    x_train = np.concatenate((mlp_oof_train, rf_oof_train, et_oof_train, svm_oof_train), axis=1)
    x_test = np.concatenate((mlp_oof_test, rf_oof_test, et_oof_test, svm_oof_test), axis=1)

    return x_train, y_train, x_test

