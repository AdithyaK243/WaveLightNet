import numpy as np
import pywt,cv2
import matplotlib.pyplot as plt
import numpy as np
import os 
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
from numpy import mean, absolute


def rotmat2aa(rotmats):
    """
    Convert rotation matrices to angle-axis using opencv's Rodrigues formula.
    Args:
        rotmats: A np array of shape (..., 3, 3)

    Returns:
        A np array of shape (..., 3)
    """
    assert rotmats.shape[-1] == 3 and rotmats.shape[-2] == 3 and len(rotmats.shape) >= 3, 'invalid input dimension'
    orig_shape = rotmats.shape[:-2]
    rots = np.reshape(rotmats, [-1, 3, 3])
    aas = np.zeros([rots.shape[0], 3])
    for i in range(rots.shape[0]):
        aas[i] = np.squeeze(cv2.Rodrigues(rots[i])[0])
    return np.reshape(aas, orig_shape + (3,)) 

def featureExtract(img, level):
  c = pywt.wavedec2(img,'haar',mode='periodization', level=level)
  if level == 3:
    cA3 = c[0]
    (cH3,cV3,cD3) = c[-3]
    (cH2,cV2,cD2) = c[-2]
    (cH1,cV1,cD1) = c[-1]
    result = np.stack((cH3, cV3, cD3), axis = -1)
  elif level == 2:
    cA2 = c[0]
    (cH2,cV2,cD2) = c[-2]
    (cH1,cV1,cD1) = c[-1]
    result = np.stack((cH2, cV2, cD2), axis = -1)
  elif level == 1:
    cA1 = c[0]
    (cH1,cV1,cD1) = c[-1]
    result = np.stack((cH1, cV1, cD1), axis = -1)
  elif level == 4:
    cA4 = c[0]
    (cH4,cV4,cD4) = c[-4]
    (cH3,cV3,cD3) = c[-3]
    (cH2,cV2,cD2) = c[-2]
    (cH1,cV1,cD1) = c[-1]
    result = cD4

  return result

def get_pose(folder, img):
  path = 'Linemod/poses'
  f_path = os.path.join(path, folder)
  if img[5:9].lstrip("0") != '':
    i = 'pose' + img[5:9].lstrip("0") + '.txt'
  else:
    i = 'pose0' + '.txt'
  
  f_path = os.path.join(f_path, i)
  with open(f_path) as f: 
      l = f.readlines()
      l0 = [float(ele) for ele in l[0].strip().split(' ') if ele.strip()]
      l1 = [float(ele) for ele in l[1].strip().split(' ') if ele.strip()]
      l2 = [float(ele) for ele in l[2].strip().split(' ') if ele.strip()]
      R = np.array([[l0[0], l0[1], l0[2]],
                    [l1[0], l1[1], l1[2]],
                    [l2[0], l2[1], l2[2]]
                    ])
      T = np.array([l0[3], l1[3], l2[3]])

      R = np.expand_dims(R, axis=0)
      R1 = rotmat2aa(R) 
      pose_6D = [T[0], T[1], T[2], R1[0][0], R1[0][1], R1[0][2]] 

      return pose_6D

def training(Dir,X, y, size):
  for folder in os.listdir(Dir):
    f_dir = os.path.join(Dir, folder)
    for img in os.listdir(f_dir):
      i = img
      path = os.path.join(f_dir,img)
      img = cv2.imread(path)
      # img = featureExtract(img,1)
      img = cv2.resize(img, (size,size))
      X[folder].append(np.array(img))
      pose = get_pose(folder, i)
      y[folder].append(pose)

  return X, y

def linemod_loader(data_shuffle=False, case=True):
    X = {}
    y = {}

    size = 56

    path = 'Linemod/real'
    for folder in os.listdir(path):
      y[folder] = []
      X[folder] = []

    training(path, X, y, size)

    X1 = []
    y1 = []
    y_test = {}
    X_test = {}
    for key, value in X.items():
      l = len(value)
      l_90 = int(0.9*l)
      X11 = value[:l_90]
      X12 = value[l_90:]
      X1 = X1 + X11
      X_test[key] = X12

    for key, value in y.items():
      l = len(value)
      l_90 = int(0.9*l)
      y11 = value[:l_90]
      y12 = value[l_90:]
      y1 = y1 + y11
      y_test[key] = y12

    #the rgb values of each pixel is reduced from 0->255 to 0->1 for easy handling
    X1 = np.array(X1)
    X1 = X1/255
    for key, value in X_test.items():
      value = np.array(value)
      value = value/255
      X_test[key] = value

    y1 = np.array(y1)

    #splitting the data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X1, y1, test_size=0.1, shuffle=data_shuffle, random_state=1)

    return X_train, X_val, y_train, y_val, X_test, y_test

def linemod_plot(history2):
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


def linemod_results(cnnLstm, X_test, y_test):
    y_pred = {}
    for key, value in X_test.items():
      y_pred[key] = cnnLstm.predict(value)

    mse_meanTx = 0
    mse_meanTy = 0
    mse_meanTz = 0
    mse_meanRx = 0
    mse_meanRy = 0 
    mse_meanRz = 0 
    Tx = []
    Ty = []
    Tz = []
    Rx = []
    Ry = []
    Rz = []
    for key, value1 in y_test.items():
      value2 = y_pred[key]
      value1 = np.array(value1)
      value2 = np.array(value2)
      print(key)
      mse_meanTx += metrics.mean_squared_error(value1[:, 0], value2[:,0])/15
      print('Tx_error', metrics.mean_squared_error(value1[:, 0], value2[:,0]))
      Tx.append(metrics.mean_squared_error(value1[:, 0], value2[:,0]))
      mse_meanTy += metrics.mean_squared_error(value1[:, 1], value2[:,1])/15
      print('Ty_error', metrics.mean_squared_error(value1[:, 1], value2[:,1]))
      Ty.append(metrics.mean_squared_error(value1[:, 1], value2[:,1]))
      mse_meanTz += metrics.mean_squared_error(value1[:, 2], value2[:,2])/15
      print('Tz_error', metrics.mean_squared_error(value1[:, 2], value2[:,2]))
      Tz.append(metrics.mean_squared_error(value1[:, 2], value2[:,2]))
      mse_meanRx += metrics.mean_squared_error(value1[:, 3], value2[:,3])/15
      print('Rx_error', metrics.mean_squared_error(value1[:, 3], value2[:,3]))
      Rx.append(metrics.mean_squared_error(value1[:, 3], value2[:,3]))
      mse_meanRy += metrics.mean_squared_error(value1[:, 4], value2[:,4])/15
      print('Ry_error', metrics.mean_squared_error(value1[:, 4], value2[:,4]))
      Ry.append(metrics.mean_squared_error(value1[:, 4], value2[:,4]))
      mse_meanRz += metrics.mean_squared_error(value1[:, 5], value2[:,5])/15
      print('Rz_error', metrics.mean_squared_error(value1[:, 5], value2[:,5]))
      Rz.append(metrics.mean_squared_error(value1[:, 5], value2[:,5]))

    m_devTx = mean(absolute(Tx - mean(Tx)))
    m_devTy = mean(absolute(Ty - mean(Ty)))
    m_devTz = mean(absolute(Tz - mean(Tz)))
    m_devRx = mean(absolute(Rx - mean(Rx)))
    m_devRy = mean(absolute(Ry - mean(Ry)))
    m_devRz = mean(absolute(Rz - mean(Rz)))

    print(mse_meanTx, '|', m_devTx)
    print(mse_meanTy, '|', m_devTy)
    print(mse_meanTz, '|', m_devTz)
    print(mse_meanRx, '|', m_devRx)
    print(mse_meanRy, '|', m_devRy)
    print(mse_meanRz, '|', m_devRz)