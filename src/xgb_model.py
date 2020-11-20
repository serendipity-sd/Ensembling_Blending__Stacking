import pandas as pd
import glob
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb


from scipy.optimize import fmin
from functools import partial


def run_training(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)


    xtrain = train_df[['lr_pred', 'rf_svd_pred', 'lr_cnt_pred']].values
    xvalid = valid_df[['lr_pred', 'rf_svd_pred', 'lr_cnt_pred']].values

    scl = StandardScaler()
    xtrain = scl.fit_transform(xtrain)
    xvalid = scl.transform(xvalid)


    clf = xgb.XGBClassifier()
    clf.fit(xtrain, train_df.sentiment.values)
    preds = clf.predict_proba(xvalid)[:, 1]
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"{fold}, {auc}")

    valid_df.loc[:, "xgb_pred"] = preds
    return valid_df




if __name__ == "__main__":
    files = glob.glob('/home/beast/PycharmProjects/abhishek_thakur/Ensembling_Blending_&_Stacking/model_preds/*.csv')
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on='id', how='left')
    # print(df.head(10))
    targets = df.sentiment.values
    pred_cols = ['lr_pred', 'rf_svd_pred', 'lr_cnt_pred']

    dfs = []
    for j in range(5):
        temp_df = run_training(df, j)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    print(metrics.roc_auc_score(fin_valid_df.sentiment.values,fin_valid_df.xgb_pred.values))