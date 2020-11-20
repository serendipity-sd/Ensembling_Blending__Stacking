import pandas as pd
from sklearn import model_selection

#if __name__ == 'main.py':
df = pd.read_csv("/home/beast/PycharmProjects/abhishek_thakur/Ensembling_Blending_&_Stacking/input/labeledTrainData.tsv", sep="\t")
print("1")
df.loc[:, 'kfold'] = -1
df = df.sample(frac=1).reset_index(drop= True)


y=df.sentiment.values

skf = model_selection.StratifiedKFold(n_splits=5)
print("2")
for f, (t_, v_) in enumerate(skf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

print("3")
df.to_csv('/home/beast/PycharmProjects/abhishek_thakur/Ensembling_Blending_&_Stacking/input/train_folds.csv', index = False)
print("4")