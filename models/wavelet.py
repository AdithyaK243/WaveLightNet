import pywt,cv2
import numpy as np

'''
Wavelet Feature Extraction
'''
def featureExtract(img, level):
    '''
        Method to create the 4 wavelet feature cA(Approxiation coefficients), cH(Horizontal coefficients), cV(Vertical coefficients), cD(Diagonal Coefficients)
        based on the number of levels

        Input: Takes the single channel image from which features have to be extracted (w_i*h_i*1)
        Output: Returns a 3 channel image by stacking cH, cV, cD decompositions (w_o*h_o*3) 
    '''
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