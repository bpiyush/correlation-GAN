import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold

from sklearn.metrics import f1_score, roc_auc_score, silhouette_score, mutual_info_score

import warnings
warnings.filterwarnings("ignore")


def classifier_evaluation(x_train, y_train, x_test, y_test):
    
    models = {"LR": LogisticRegression(), "RF": RandomForestClassifier(), "GB": GradientBoostingClassifier()}

    results = {}
    for clf_name in models:
        clf = models[clf_name]
        
        # prepare the training data
        X = pd.concat([x_train, x_test])
        X = pd.get_dummies(X)
        
        Y = pd.concat([y_train, y_test])
        label_encoder = LabelEncoder().fit(Y)
        Y = label_encoder.transform(Y)

        # fit the model
        clf.fit(X.values[:x_train.shape[0]], Y[:x_train.shape[0]])
        
        # predict on test data
        y_pred = clf.predict(X.values[x_train.shape[0]:])
        y_scores = clf.predict_proba(X.values[x_train.shape[0]:])
    
        # performace metrics
        f1 = f1_score(Y[x_train.shape[0]:], y_pred)
        roc_auc = roc_auc_score(Y[x_train.shape[0]:], y_scores[:,1])

        results[clf_name] = roc_auc
    
    return results


def utility_evaluation(X_train, X_generated, X_test, label_index, discretize_label=False):
    
    Y_ori_train = X_train[label_index]
    X_ori_train = X_train.drop([label_index], axis=1)
    
    Y_gen_train = X_generated[label_index]
    X_gen_train = X_generated.drop([label_index], axis=1)
    
    Y_ori_test = X_test[label_index]
    X_ori_test = X_test.drop([label_index], axis=1)
    
    if discretize_label:
        joint_labels = pd.concat([Y_ori_train, Y_ori_test])
        median = np.mean(joint_labels.values)
        
        Y_gen_train = (Y_gen_train > median).astype('int')
        Y_ori_train = (Y_ori_train > median).astype('int')
        Y_ori_test = (Y_ori_test > median).astype('int')

    if len(np.unique(Y_ori_train.values)) != len(np.unique(Y_gen_train.values)):
        print("Generated labels: ", np.unique(Y_gen_train.values))
        print("Original labels: ", np.unique(Y_ori_train.values))
        return

    results_ori = classifier_evaluation(X_ori_train, Y_ori_train, X_ori_test, Y_ori_test)
    results_gen = classifier_evaluation(X_gen_train, Y_gen_train, X_ori_test, Y_ori_test)
    
    return {'original': results_ori, 'generated': results_gen}
