import random,cv2
import numpy as np
import os
import pandas as pd

def noisy_spec(image):
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)*1.1
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + image * gauss
    return noisy

def make_noisyData():
    blur_list = [0.95, 0.9, 0.8, 0.85]
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
        
        permutation_idxs = np.random.permutation(len(os.listdir(folderp)))
        l = int(0.6*len(permutation_idxs))

        for i in permutation_idxs[:l]:
            img = os.listdir(folderp)[i]
            imgp = os.path.join(folderp, img)
            image = cv2.imread(imgp)
            blur = random.choice(blur_list)
            b = cv2.GaussianBlur(image,(5,5),blur)
            b = b*255
            b = cv2.resize(b, (size, size))
            if img[-5:-4] == 'i': 
                X_real_i.append(np.array(b))
                y2_i.append(img)
            elif img[-5:-4] == 'o': 
                X_real_o.append(np.array(b))
                y2_o.append(img)
            elif img[-5:-4] == 's': 
                X_real_s.append(np.array(b))
                y2_s.append(img)

        l2 = l + int(0.2*len(permutation_idxs))

        for j in permutation_idxs[l:l2]:
            img = os.listdir(folderp)[j]
            imgp = os.path.join(folderp, img)
            image = cv2.imread(imgp)
            blur = noisy_spec(image)
            b = b*255
            b = cv2.resize(b, (size, size))
            if img[-5:-4] == 'i': 
                X_real_i.append(np.array(b))
                y2_i.append(img)
            elif img[-5:-4] == 'o': 
                X_real_o.append(np.array(b))
                y2_o.append(img)
            elif img[-5:-4] == 's': 
                X_real_s.append(np.array(b))
                y2_s.append(img)
        
        for k in permutation_idxs[l2:]:
            img = os.listdir(folderp)[k]
            imgp = os.path.join(folderp, img)
            image = cv2.imread(imgp)
            b = image
            b = cv2.resize(b, (size, size))
            if img[-5:-4] == 'i': 
                X_real_i.append(np.array(b))
                y2_i.append(img)
            elif img[-5:-4] == 'o': 
                X_real_o.append(np.array(b))
                y2_o.append(img)
            elif img[-5:-4] == 's': 
                X_real_s.append(np.array(b))
                y2_s.append(img)
            # p = d_path+ folder + '/'+img[-10:]
            # cv2.imwrite(p, image)

    p_real = ['Data/Real_Dataset/pose.csv']  

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

    return X_test_i, X_test_o, X_test_s, y_test_i, y_test_o, y_test_s