import argparse
import measures
import preprocess
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import zero_one_loss
from tqdm import tqdm
import numpy as np
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import average_precision_score

def noisyfeature(X,num):
    a = X.shape[0]  # X的行数
    X1 = np.random.normal(0,5,(a,num))  # 返回一个指定形状的数组，数组中的值服从标准正态分布（均值为0，方差为5）
    X2 = np.concatenate([X, X1], 1)
    return X2

def sucrate(idx,t):
    suc = 0
    for i in idx[0:t]:
        if i<t:
            suc = suc+1
    return suc/t

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, 
                        default="C:/Users/lihai/Desktop/MLFS/data")
    parser.add_argument("--save_path", type=str, 
                        default="C:/Users/lihai/Desktop/MLFS/num_feature_result")
    parser.add_argument("--data_names", type=list, 
                        default=["emotions", "image", "scene", "yeast"])
    parser.add_argument("--data_dict", type=dict,
                        default={"emotions": {"feature": 72, "label": 6},
                                 "image": {"feature": 294, "label": 5},
                                 "scene": {"feature": 294, "label": 6}, 
                                 "yeast": {"feature": 103, "label": 14}})
    parser.add_argument("--parameter", type=dict,
                        default={
                            "emotions": {
                                "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2], 
                                "beta": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2], 
                                "rho": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                "walk_length": 50},
                            "image": {
                                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1], 
                                "beta": [1, 1e1, 1e2], 
                                "rho": [0.0, 0.3],
                                "walk_length": 50},
                            "scene": {
                                "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1], 
                                "beta": [1, 1e1, 1e2], 
                                "rho": [0.0, 0.3, 0.7, 0.8],
                                "walk_length": 50},
                            "yeast": {
                                "alpha": [1e-3, 1e-2, 1e-1, 1], 
                                "beta": [1, 1e1, 1e2, 1e3], 
                                "rho": [0.0, 0.6, 0.8],
                                "walk_length": 50}})
    return parser.parse_args()


def pre_process(path, data_dict, name):
    processor = preprocess.PreProcess()
    data_train = pd.read_csv(f"{path}/{name}/{name}-train.csv", delimiter=",")
    data_test = pd.read_csv(f"{path}/{name}/{name}-test.csv", delimiter=",")
    x_train = data_train.iloc[:, :data_dict[name]["feature"]].values
    x_test = data_test.iloc[:, :data_dict[name]["feature"]].values
    y_train = data_train.iloc[:, data_dict[name]["feature"]:].values
    y_test = data_test.iloc[:, data_dict[name]["feature"]:].values
    x_train, y_train = processor.nonsense_treat(x_train, y_train)
    x_test, y_test = processor.nonsense_treat(x_test, y_test)
    return x_train, y_train, x_test, y_test


def laplacian(A):
    #A是相似度矩阵S
    n = A.shape[0]
    P = np.eye(n)
    A = (A + A.T) / 2
    for i in range(n):
        P[i, i] = np.sum(A[i, :])
    return P - A,P # L,A(度矩阵A)


def feature_selection(W, num_fea):
    feature_score = np.sqrt(np.sum(W ** 2, axis=1))
    return np.argsort(feature_score)[-num_fea:]


def evaluate(y_true, y_pre):
    MLM = measures.MultiLabelMetrics(y_true, y_pre)
    RM = measures.RankingMetrics(y_true, y_pre)
    HL = round(MLM.hamming_loss(), 4)
    RL=label_ranking_loss(y_true, y_pre)
    #RL = RM.ranking_loss()
    OE = round(RM.one_error(), 4)
    Cov = round(RM.coverage(), 4)
    #AP = round(RM.average_precision(), 4)
    AP = average_precision_score(y_true, y_pre)
    # macro = round(MLM.macro_F1(), 4)
    # micro = round(MLM.micro_F1(), 4)

    macro = round(f1_score(y_true, y_pre, average='macro'),4)
    micro = round(f1_score(y_true, y_pre, average='micro'),4)

    return np.array([HL, RL, OE, Cov, AP,macro,micro])
    #return np.array([HL, RL, OE, Cov, AP, macro, micro,'|||'])
    #return np.array([OE])
    # return [HL, RL, OE, Cov, AP]

def evaluate2(y_true, y_pre):
    MLM = measures.MultiLabelMetrics(y_true, y_pre)
    RM = measures.RankingMetrics(y_true, y_pre)
    HL = round(MLM.hamming_loss(), 4)
    RL=label_ranking_loss(y_true, y_pre)
    #RL = RM.ranking_loss()
    OE = round(RM.one_error(), 4)
    Cov = round(RM.coverage(), 4)
    #AP = round(RM.average_precision(), 4)
    AP = average_precision_score(y_true, y_pre)
    macro = round(MLM.macro_F1(), 4)
    micro = round(MLM.micro_F1(), 4)

    #macro = round(f1_score(y_true, y_pre, average='macro'),4)
    #micro = round(f1_score(y_true, y_pre, average='micro'),4)

    #return np.array([HL, RL, OE, Cov, AP,macro,micro])
    return np.array([HL, RL, OE, Cov, AP, macro, micro,'|||'])



