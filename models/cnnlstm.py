import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Flatten,Dropout
from keras.layers import MaxPooling2D, LSTM
from keras.layers import Dense
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Dense
import keras.backend as K
from keras.layers import LeakyReLU

def reg_model(inp_size, model_name = 'CnnLstm'):
    '''
        Model function to create the CNN-LSTM network to regress the 6DOF pose vector

        Input: Takes the image size after the 3 stage wavelet decomposition
        Output: Returns the model
    '''
    inp = Input(inp_size)
    conv1 = Conv2D(16, (3, 3), padding='same',activation = 'relu',kernel_initializer='normal', input_shape = inp_size)(inp)
    pool1 = MaxPooling2D()(conv1)
    drop1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(16, (3, 3), padding='same',activation = 'relu',kernel_initializer='normal')(drop1)
    pool2 = MaxPooling2D()(conv2)
    drop2 = Dropout(0.25)(pool2)

    conv3 = Conv2D(32, (3, 3), padding='same',activation = 'relu',kernel_initializer='normal')(drop2)
    pool3 = MaxPooling2D()(conv3)
    pool3 = Dropout(0.25)(pool3)

    out_e = tf.reshape(pool3, (-1, 49, 32))
    if model_name=='CnnLstm':
        lstm = LSTM(128,return_sequences=True,input_shape = (49, 32))(out_e)
        flat = Flatten()(lstm)
        dense1 = Dense(64)(flat)
        dense2 = Dense(6)(dense1)
        out = LeakyReLU(alpha=0.1)(dense2)
    elif model_name=='Cnn':
        flat = Flatten()(out_e)
        dense1 = Dense(64)(flat)
        dense2 = Dense(6)(dense1)
        out = LeakyReLU(alpha=0.1)(dense2)
        
    model = Model(inputs = inp, outputs=out)
    print(model.summary())

    return model