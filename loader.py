import numpy as np
import pywt,cv2
import matplotlib.pyplot as plt
import numpy as np
import os 
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import keras.backend as K
from keras.layers import LeakyReLU

'''
Wavelet Feature Extraction
'''
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

#training values
inch_train = 'Data/in/inchworm'
omega_train = 'Data/in/omega'
std_train = 'Data/in/standard'
original = 'Data/Real_Dataset/input_images'

size = 56

def training(label, Dir, X1, y1, case):
    for img in os.listdir(Dir):
      i = img
      path = os.path.join(Dir,img)
      if case:
        img = cv2.imread(path, 0)
        img = featureExtract(img,3)
      else:
        img = cv2.imread(path)
      img = cv2.resize(img, (size,size))
      X1.append(np.array(img))
      y1[label].append(i)

def training2(Dir, X_real_i, y2_i, X_real_o, y2_o, X_real_s, y2_s, case):
    for img in os.listdir(Dir):
        i = img
        path = os.path.join(Dir,img)
        img = cv2.imread(path)
        img = cv2.resize(img, (size, size))
        if i[-5:-4] == 'i': 
          X_real_i.append(np.array(img))
          y2_i.append(i)
        elif i[-5:-4] == 'o': 
          X_real_o.append(np.array(img))
          y2_o.append(i)
        elif i[-5:-4] == 's': 
          X_real_s.append(np.array(img))
          y2_s.append(i)


def train_data_loader(data_shuffle = False, case=True):      
    X1 = []
    X_real_i = []
    X_real_o = []
    X_real_s = []
    y1 = {'inchworm':[], 'omega':[], 'standard':[]}
    y2_i = []
    y2_o = []
    y2_s = []
    X_test_i = []
    y_test_i = []
    X_test_o = []
    y_test_o = []
    X_test_s = []
    y_test_s = []

    Y1 = []
    y_real_i = []
    y_real_o = []
    y_real_s = []
    training('inchworm', inch_train, X1, y1, case)
    print((len(X1)))
    training('omega',omega_train, X1, y1, case)
    print(len(X1))
    training('standard',std_train, X1, y1, case)
    print(len(X1))

    training2(original, X_real_i, y2_i, X_real_o, y2_o, X_real_s, y2_s)

    p = {'inchworm':'Data/inchworm_pose.csv', 'omega':'Data/omega_pose.csv','standard':'Data/standard_pose.csv'}
    p_real = ['Data/Real_Dataset/pose.csv']  
 
    for lab, pose_path in p.items():
        pose = pd.read_csv(pose_path)
        for j in y1[lab]:
          for i in  range(4000):
            if pose['Img_Name'][i] == j: Y1.append([pose['Tx'][i], pose['Ty'][i], pose['Tz'][i], pose['Rx'][i], pose['Ry'][i], pose['Rz'][i]])
    for pose_path in p_real:
        pose = pd.read_csv(pose_path)
        for j in y2_i:
          for i in range(100):
            if pose['Img_Name'][i] == j : y_real_i.append([pose['Tx'][i], pose['Ty'][i], pose['Tz'][i], pose['Rx'][i], pose['Ry'][i], pose['Rz'][i]])
        for j in y2_o:
          for i in range(100,200):  
            if pose['Img_Name'][i] == j: y_real_o.append([pose['Tx'][i], pose['Ty'][i], pose['Tz'][i], pose['Rx'][i], pose['Ry'][i], pose['Rz'][i]])
        for j in y2_s:
          for i in range(200,300):  
            if pose['Img_Name'][i] == j: y_real_s.append([pose['Tx'][i], pose['Ty'][i], pose['Tz'][i], pose['Rx'][i], pose['Ry'][i], pose['Rz'][i]])
    Y1 = np.array(Y1)
    y_real_i = np.array(y_real_i)
    y_real_o = np.array(y_real_o)
    y_real_s = np.array(y_real_s)

    print(y_real_i.shape)
    print(y_real_o.shape)
    print(y_real_s.shape)

    y_test_i = y_real_i
    y_test_o = y_real_o
    y_test_s = y_real_s

    #the rgb values of each pixel is reduced from 0->255 to 0->1 for easy handling
    X1 = np.array(X1)
    X1 = X1/255
    X_test_i = np.array(X_real_i)
    X_test_o = np.array(X_real_o)
    X_test_s = np.array(X_real_s)
    X_test_i = X_test_i/255
    X_test_o = X_test_o/255
    X_test_s = X_test_s/255


    X1 = np.expand_dims(X1,axis=-1)
    X_test_i = np.expand_dims(X_test_i,axis = -1)
    X_test_o = np.expand_dims(X_test_o,axis = -1)
    X_test_s = np.expand_dims(X_test_s,axis = -1)

    #splitting the data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X1, Y1, test_size=0.1, shuffle=data_shuffle, random_state=1)

    return X_train, X_val, y_train, y_val, X_test_i, X_test_o, X_test_s, y_test_i, y_test_o, y_test_s

def test_data_loader(case=True):      
    X_real_i = []
    X_real_o = []
    X_real_s = []
    y2_i = []
    y2_o = []
    y2_s = []
    X_test_i = []
    y_test_i = []
    X_test_o = []
    y_test_o = []
    X_test_s = []
    y_test_s = []

    y_real_i = []
    y_real_o = []
    y_real_s = []

    training2(original, X_real_i, y2_i, X_real_o, y2_o, X_real_s, y2_s, case)

    p_real = ['Data/Real_Dataset/pose.csv']  
    
 
    for pose_path in p_real:
        pose = pd.read_csv(pose_path)
        for j in y2_i:
          for i in range(100):
            if pose['Img_Name'][i] == j : y_real_i.append([pose['Tx'][i], pose['Ty'][i], pose['Tz'][i], pose['Rx'][i], pose['Ry'][i], pose['Rz'][i]])
        for j in y2_o:
          for i in range(100,200):  
            if pose['Img_Name'][i] == j: y_real_o.append([pose['Tx'][i], pose['Ty'][i], pose['Tz'][i], pose['Rx'][i], pose['Ry'][i], pose['Rz'][i]])
        for j in y2_s:
          for i in range(200,300):  
            if pose['Img_Name'][i] == j: y_real_s.append([pose['Tx'][i], pose['Ty'][i], pose['Tz'][i], pose['Rx'][i], pose['Ry'][i], pose['Rz'][i]])
    y_real_i = np.array(y_real_i)
    y_real_o = np.array(y_real_o)
    y_real_s = np.array(y_real_s)

    print(y_real_i.shape)
    print(y_real_o.shape)
    print(y_real_s.shape)

    y_test_i = y_real_i
    y_test_o = y_real_o
    y_test_s = y_real_s

    #the rgb values of each pixel is reduced from 0->255 to 0->1 for easy handling
    X_test_i = np.array(X_real_i)
    X_test_o = np.array(X_real_o)
    X_test_s = np.array(X_real_s)
    X_test_i = X_test_i/255
    X_test_o = X_test_o/255
    X_test_s = X_test_s/255


    X_test_i = np.expand_dims(X_test_i,axis = -1)
    X_test_o = np.expand_dims(X_test_o,axis = -1)
    X_test_s = np.expand_dims(X_test_s,axis = -1)

    return X_test_i, X_test_o, X_test_s, y_test_i, y_test_o, y_test_s