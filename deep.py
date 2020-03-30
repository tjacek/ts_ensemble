import keras
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras.layers.normalization import BatchNormalization

def make_conv(params):
    input_layer = Input(shape=(params['ts_len'], params['n_feats'],1))
    activ='relu' #'elu'
    pool1=add_conv_layer(input_layer,0,activ=activ)
    pool2=add_conv_layer(pool1,1,activ=activ)
    kernel_regularizer=regularizers.l1(0.001)
    hidden_layer = Dense(100,name='hidden', activation=activ,
                         kernel_regularizer=kernel_regularizer)(Flatten()(pool2))
   
    hidden_layer=BatchNormalization()(hidden_layer)
    drop1=Dropout(0.5)(hidden_layer)
    output_layer = Dense(units=params['n_cats'], activation='softmax')(drop1)
    model=Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True),
             metrics=['accuracy'])
    model.summary()
    return model

def add_conv_layer(input_layer,i=0,n_kerns=16,activ='relu',
                    kern_size=(8,1),pool_size=(4,1)):
    i=str(i)
    conv1=Conv2D(n_kerns, kernel_size=kern_size,
            activation=activ,name='conv'+i)(input_layer)
    pool1=MaxPooling2D(pool_size=pool_size,name='pool'+i)(conv1)
    return pool1