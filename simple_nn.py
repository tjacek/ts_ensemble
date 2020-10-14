import keras
from keras.layers import Input,Dense,Dropout
from keras.models import Model
import files

def train_model(in_path):
    feat_dict=files.get_feats(in_path)
    sample_size=feat_dict.dim()[0]
    basic_model(sample_size,n_dense=64,n_cats=20)

def basic_model(sample_size,n_dense=64,n_cats=20):
    input_img = Input(shape=(sample_size,))
    x=input_img
    x=Dropout(0.5)(x)
    x=Dense(n_dense, activation='relu',name="hidden")(x)#,kernel_regularizer=regularizers.l1(0.01),)(x)
    x=Dense(units=n_cats,activation='softmax')(x)
    model = Model(input_img, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=0.001))
    model.summary() 
    return model

train_model("out/max_z/dtw")