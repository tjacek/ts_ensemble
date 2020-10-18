import keras
from keras.layers import Input,Dense,Dropout
from keras.models import Model
import files

def extract(in_path,nn_path,out_path):
    feat_dict=files.get_feats(in_path)
    model = keras.models.load_model(nn_path)
    extractor=Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)
    X,y,names=feat_dict.as_dataset()
    new_X=extractor.predict(X)
    files.save_feats(new_X,names,out_path)

def train_model(in_path,out_path=None,n_epochs=300,binary=None):
    feat_dict=files.get_feats(in_path)
    sample_size=feat_dict.dim()[0]
    train,test=feat_dict.split()
    X,y,name=train.as_dataset()
    n_cats=20
    if(not binary is None):
        if(type(binary)==int):
            binary=[binary]
        binary={ binary_i:i+1 for i,binary_i in enumerate(binary)}
        def helper(y_i):
            if( y_i in binary):
                return binary[y_i]
            return 0
        y=[ helper(y_i) for y_i in y]
        n_cats=len(binary)+1
    y=keras.utils.to_categorical(y)
    model=basic_model(sample_size,n_dense=64,n_cats=n_cats)
    model.fit(X,y,epochs = n_epochs)
    if(out_path):
        model.save(out_path)
    return model	

def basic_model(sample_size,n_dense=64,n_cats=20):
    input_img = Input(shape=(sample_size,))
    x=input_img
    x=Dense(n_dense, activation='relu',name="hidden")(x)#,kernel_regularizer=regularizers.l1(0.01),)(x)
    x=Dense(units=n_cats,activation='softmax')(x)
    model = Model(input_img, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=0.0001))
    model.summary() 
    return model

train_model("full/dtw","full/nn",binary=[3,6,7])
extract("full/dtw","full/nn","full/deep2")