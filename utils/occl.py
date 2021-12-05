import cv2
import os
import numpy as np
import random
import pandas as pd

def CutOut(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    mask = np.ones(img.shape)
    mask[y:y+height, x:x+width, :] = 0
    img = img * mask
    img =cv2.resize(img, (56, 56))
    return img
    
def add_occlusion(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  cutout_image_1 = CutOut(img, 20, 20)

  return cutout_image_1

def make_occlData():
    path = 'Data/Real_Dataset/in'
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

    size = 56
    for folder in os.listdir(path):
        folderp = os.path.join(path, folder)

        for k in range(len(os.listdir(folderp))):
            img = os.listdir(folderp)[k]
            imgp = os.path.join(folderp, img)
            b = add_occlusion(imgp)
            if img[-5:-4] == 'i': 
                X_real_i.append(np.array(b))
                y2_i.append(img)
            elif img[-5:-4] == 'o': 
                X_real_o.append(np.array(b))
                y2_o.append(img)
            elif img[-5:-4] == 's': 
                X_real_s.append(np.array(b))
                y2_s.append(img)

    p_real = ['/content/Data/Real_Dataset/pose.csv']  

    y_real_i = []
    y_real_o = []
    y_real_s = []
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