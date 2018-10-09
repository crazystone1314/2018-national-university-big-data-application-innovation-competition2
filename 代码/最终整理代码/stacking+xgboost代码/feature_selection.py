# -*- coding:utf-8 -*-


import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


def get_top_n_features(train_data_X, train_data_Y, train_data_columns, top_n_features):
    '''
    利用不同的模型来对特征进行筛选，选出较为重要的特征
    :param train_data_X:
    :param train_data_Y:
    :param train_data_columns:
    :param top_n_features:
    :return:
    '''
    # randomforest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame(
        {'feature': list(train_data_columns), 'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Feeatures from RF Classifier')
    print(str(features_top_n_rf[:10]))

    # AdaBoost
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(train_data_columns),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Features from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best DT Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(train_data_columns),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(train_data_columns),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Bset DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(train_data_columns),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))

    # merge the three models
    features_top_n = pd.concat(
        [features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt],
        ignore_index=True).drop_duplicates()
    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                     feature_imp_sorted_gb, feature_imp_sorted_dt], ignore_index=True)

    return features_top_n, features_importance



def create_traindata_and_testdata(train_data, test_data):
    '''
    依据筛选出的特征构建训练集和测试集
    :param train_data:
    :param test_data:
    :return:
    '''
    train_data_X = train_data.drop(['hand'], axis=1)
    train_data_Y = train_data.hand

    #筛选每个模型认为最重要的前N个特征
    feature_to_pick = 20
    train_data_columns = train_data_X.columns
    feature_top_n, feature_importance = get_top_n_features(train_data_X, train_data_Y, train_data_columns,
                                                           feature_to_pick)
    train_data_X = train_data_X[feature_top_n]
    test_data_X = test_data[feature_top_n]
    importance_feature = pd.DataFrame(feature_top_n.values, columns=['importance_feature'])
    #将筛选的特征保存到文件当中
    importance_feature.to_csv('dataset/importance_feature_data.csv', index=False)

    return train_data_X, train_data_Y, test_data_X