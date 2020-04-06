import numpy as np
import keras.utils
from sklearn.metrics import accuracy_score,classification_report
from keras.models import Model
import deep,files

def simple_exp(in_path,n_epochs=1000):
    seq_dict=files.get_seqs(in_path)
    train,test=files.split(seq_dict)
    X,y,names,params=prepare_data(train)
    model=deep.make_conv(params)
    model.fit(X,y,epochs = n_epochs)
#    test_model(model,test)
    extrac_feats(model,seq_dict,"feats.txt")

def prepare_data(seq_dict):
    names=seq_dict.keys()
    data=list(seq_dict.values())
    X=np.array(data)
    X=np.expand_dims(X,axis=-1)
    y=[ int(name_i.split("_")[0])-1 for name_i in names] 
    params={'ts_len':X.shape[1], 'n_feats': X.shape[2],
                'n_cats':max(y)+1}
    y=keras.utils.to_categorical(y)
    return X,y,names,params

def test_model(model,test):
    X,y_true,names,params=prepare_data(test)
    y_pred=model.predict(X)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    report=classification_report(y_pred,y_true,digits=4)
    print(report)

def extrac_feats(model,seq_dict,out_path):
    extractor=Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)
    X,y,names,params=prepare_data(seq_dict)
    X=extractor.predict(X) 
#    new_seqs={ name_i:extractor.predict(X[i]) 
#        for i,name_i in enumerate(names)}
    files.save_feats(X,names,out_path)

simple_exp("agum")