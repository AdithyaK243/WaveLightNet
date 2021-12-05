import os
import re
from keras import models
import numpy as np
import argparse

from loader import *
from models.cnnlstm import *
from models.wavelet import *

from tensorflow import keras
from numpy import mean, absolute
import pandas as pd
from tensorflow.keras.optimizers import SGD
from datetime import datetime
import sklearn.metrics as metrics

def results(cnnLstm, X_test_i, X_test_o, X_test_s, y_test_i, y_test_o, y_test_s):
    y_pred_i = cnnLstm.predict(X_test_i)
    y_pred_o = cnnLstm.predict(X_test_o)
    y_pred_s = cnnLstm.predict(X_test_s)

    mae_i = metrics.mean_absolute_error(y_test_i, y_pred_i)
    mae_o = metrics.mean_absolute_error(y_test_o, y_pred_o)
    mae_s = metrics.mean_absolute_error(y_test_s, y_pred_s)

    mse_i = metrics.mean_squared_error(y_test_i, y_pred_i) 
    mse_o = metrics.mean_squared_error(y_test_o, y_pred_o) 
    mse_s = metrics.mean_squared_error(y_test_s, y_pred_s) 

    mse_i_Tx = metrics.mean_squared_error(y_test_i[:, 0], y_pred_i[:,0])
    mse_i_Ty = metrics.mean_squared_error(y_test_i[:, 1], y_pred_i[:,1])
    mse_i_Tz = metrics.mean_squared_error(y_test_i[:, 2], y_pred_i[:,2])
    mse_i_Rx = metrics.mean_squared_error(y_test_i[:, 3], y_pred_i[:,3])
    mse_i_Ry = metrics.mean_squared_error(y_test_i[:, 4], y_pred_i[:,4])
    mse_i_Rz = metrics.mean_squared_error(y_test_i[:, 5], y_pred_i[:,5])

    mse_o_Tx = metrics.mean_squared_error(y_test_o[:, 0], y_pred_o[:,0])
    mse_o_Ty = metrics.mean_squared_error(y_test_o[:, 1], y_pred_o[:,1])
    mse_o_Tz = metrics.mean_squared_error(y_test_o[:, 2], y_pred_o[:,2])
    mse_o_Rx = metrics.mean_squared_error(y_test_o[:, 3], y_pred_o[:,3])
    mse_o_Ry = metrics.mean_squared_error(y_test_o[:, 4], y_pred_o[:,4])
    mse_o_Rz = metrics.mean_squared_error(y_test_o[:, 5], y_pred_o[:,5])

    mse_s_Tx = metrics.mean_squared_error(y_test_s[:, 0], y_pred_s[:,0])
    mse_s_Ty = metrics.mean_squared_error(y_test_s[:, 1], y_pred_s[:,1])
    mse_s_Tz = metrics.mean_squared_error(y_test_s[:, 2], y_pred_s[:,2])
    mse_s_Rx = metrics.mean_squared_error(y_test_s[:, 3], y_pred_s[:,3])
    mse_s_Ry = metrics.mean_squared_error(y_test_s[:, 4], y_pred_s[:,4])
    mse_s_Rz = metrics.mean_squared_error(y_test_s[:, 5], y_pred_s[:,5])

    mse_meanTx = (mse_i_Tx + mse_o_Tx + mse_s_Tx) / 3 
    mse_meanTy = (mse_i_Ty + mse_o_Ty + mse_s_Ty) / 3 
    mse_meanTz = (mse_i_Tz + mse_o_Tz + mse_s_Tz) / 3 
    mse_meanRx = (mse_i_Rx + mse_o_Rx + mse_s_Rx) / 3 
    mse_meanRy = (mse_i_Ry + mse_o_Ry + mse_s_Ry) / 3 
    mse_meanRz = (mse_i_Rz + mse_o_Rz + mse_s_Rz) / 3 

    from numpy import mean, absolute
    import pandas as pd

    m_devTx = mean(absolute([mse_i_Tx,mse_o_Tx, mse_s_Tx] - mean([mse_i_Tx,mse_o_Tx, mse_s_Tx])))
    m_devTy = mean(absolute([mse_i_Ty,mse_o_Ty, mse_s_Ty] - mean([mse_i_Ty,mse_o_Ty, mse_s_Ty])))
    m_devTz = mean(absolute([mse_i_Tz,mse_o_Tz, mse_s_Tz] - mean([mse_i_Tz,mse_o_Tz, mse_s_Tz])))
    m_devRx = mean(absolute([mse_i_Rx,mse_o_Rx, mse_s_Rx] - mean([mse_i_Rx,mse_o_Rx, mse_s_Rx])))
    m_devRy = mean(absolute([mse_i_Ry,mse_o_Ry, mse_s_Ry] - mean([mse_i_Ry,mse_o_Ry, mse_s_Ry])))
    m_devRz = mean(absolute([mse_i_Rz,mse_o_Rz, mse_s_Rz] - mean([mse_i_Rz,mse_o_Rz, mse_s_Rz])))

    print(mse_meanTx, '|', m_devTx)
    print(mse_meanTy, '|', m_devTy)
    print(mse_meanTz, '|', m_devTz)
    print(mse_meanRx, '|', m_devRx)
    print(mse_meanRy, '|', m_devRy)
    print(mse_meanRz, '|', m_devRz)

def plot(history2):
    plt.plot(history2.history['mse'])
    plt.plot(history2.history['val_mse'])
    plt.title('mean_squared_error')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history2.history['mae'])
    plt.plot(history2.history['val_mae'])
    plt.title('mean_absolute_error')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def main():
    '''
        Main function to set arguments
    '''

    # arguments
    parser = argparse.ArgumentParser()

    # hyper parameters
    parser.add_argument('--lr',          type=float, default = 0.01)
    parser.add_argument('--epoch',       type=int,   default = 10)
    parser.add_argument('--batch_size',  type=int,   default = 12)

    # training & testing parameters
    parser.add_argument('--case',        type=bool,  default=False)
    parser.add_argument('--data',        type=bool,  default=False)
    parser.add_argument('--train',       type=bool,  default=False)
    parser.add_argument('--test_model',  type=bool,  default=True) 
    parser.add_argument('--model_name',  type=str,   default='CnnLstm')
    parser.add_argument('--chkpt_path',  type=str,   default='checkpoints/CnnLstm_FeatExt_dataFalse_trainFalse.h5')   

    args = parser.parse_args()

    if args.test_model == True:
        model = keras.models.load_model(args.chkpt_path)
        X_test_i, X_test_o, X_test_s, y_test_i, y_test_o, y_test_s = test_data_loader(args.case)

        results(model,  X_test_i, X_test_o, X_test_s, y_test_i, y_test_o, y_test_s)
    else:
        X_train, X_val, y_train, y_val, X_test_i, X_test_o, X_test_s, y_test_i, y_test_o, y_test_s = train_data_loader(args.data, args.case)

        model = reg_model((56,56,3), args.model_name)

        #here we specify the optimizer and learning rate
        model.compile(optimizer = SGD(learning_rate = args.lr) , loss = tf.keras.losses.MeanSquaredError(),metrics=['mse', 'mae']) #Adam(learning_rate=3e-4)

        #training the data
        history2 = model.fit(X_train,y_train,batch_size = args.batch_size, epochs = args.epoch, verbose = 1,validation_data = (X_val, y_val),shuffle = args.train)

        now = datetime.now()

        # save model and architecture to single file
        model.save('checkpoints/' + now + '_' + args.model_name + 'FeatExt_' + args.case + '_data' + str(args.data) + '_train' + str(args.train) + ".h5")
        print("Saved model to disk")

        plot(history2)
        results(model,  X_test_i, X_test_o, X_test_s, y_test_i, y_test_o, y_test_s)

if __name__ == "__main__":
    main()