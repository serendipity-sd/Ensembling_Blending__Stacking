import pandas as pd
import glob
from sklearn import metrics
import numpy as np

from scipy.optimize import fmin
from functools import partial
class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        x_coef = X * coef
        predictions = np.sum(x_coef, axis = 1)
        auc_score = metrics.roc_auc_score(y,predictions)
        return -1.0* auc_score

    def fit(self, X,y):
        partial_loss = partial(self._auc, X=X, y=y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        self.coef_ = fmin(partial_loss, init_coef, disp = True)

    def predict(self, X):
        x_coef = X* self.coef_
        predictions = np.sum(x_coef,axis= 1)
        return predictions

def run_training(pred_df, fold):
        train_df = pred_df[pred_df.kfold != fold].reset_index(drop= True)
        valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

        xtrain = train_df[['lr_pred', 'rf_svd_pred', 'lr_cnt_pred']].values
        xvalid = valid_df[['lr_pred', 'rf_svd_pred', 'lr_cnt_pred']].values

        opt = OptimizeAUC()
        opt.fit(xtrain, train_df.sentiment.values)
        preds = opt.predict(xvalid)
        auc =  metrics.roc_auc_score(valid_df.sentiment.values,preds)

        valid_df.loc[:, "opt_pred"] = preds

        return  opt.coef_

        print(f"{fold}, {auc}")

if __name__=="__main__":
    print("1")
    files = glob.glob('/home/beast/PycharmProjects/abhishek_thakur/Ensembling_Blending_&_Stacking/model_preds/*.csv')
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on='id', how='left')
    #print(df.head(10))
    targets = df.sentiment.values
    pred_cols = ['lr_pred', 'rf_svd_pred', 'lr_cnt_pred']

    coefs = []
    for j in range(5):
        coefs.append(run_training(df,j))

    coefs = np.array(coefs)
    print(coefs)
    coefs = np.mean(coefs,axis = 0)
    print(coefs)

    wt_avg = (
            coefs[0] * df.lr_pred.values
            + coefs[1]*df.rf_svd_pred.values
            + coefs[2]* df.lr_cnt_pred.values
    )
    print("optimal Auc after finding AUC")
    print(metrics.roc_auc_score(targets, wt_avg))