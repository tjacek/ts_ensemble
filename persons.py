import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import files

def orth_selection(in_path):
    datasets=[ files.get_feats(path_i)
        for path_i in files.top_files(in_path)]
    acc=[ get_acc(data_i) for data_i in datasets]
    print(acc)

def get_acc(data_i):
    train_i,test_i=files.split(data_i)
    X=np.array(list(train_i.values()))
    y=[ name_i.split("_")[1] for name_i in train_i.keys()]	
    clf_i=LogisticRegression(solver='liblinear')
    clf_i.fit(X,y)
    y_pred=clf_i.predict(X)
    acc_i=accuracy_score(y,y_pred)
    return acc_i


orth_selection("../MSR/sub_feats")