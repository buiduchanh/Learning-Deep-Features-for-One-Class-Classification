from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
# dataset
import os
import glob

def kyocera_data(data_path):

    #typedata : numpy array
    #typelabel : [0 0 0 1] #normal_cp #abnormal_cp #normal_smd #abnormal_smd
    x_train_s, x_test_s, x_test_b = [], [], []
    x_ref, y_ref = [], []
    x_test_s_path , x_test_b_path =[] ,[]
    cp1_path = os.path.join(data_path, 'kyocera_CP1')
    smd_path = os.path.join(data_path, 'kyocera_SMD')

    #make reference data
    cp1_normal_path = os.path.join(cp1_path, 'train', 'OK')
    cp1_normal = sorted(glob.glob('{}/*'.format(cp1_normal_path)))
    for cp1 in cp1_normal:
        x_train_s.append(cp1)
        x_ref.append(cp1)
        y_ref.append(np.array([1, 0]))
    # #make test data
    cp1_test_path =  os.path.join(cp1_path, 'test')
    cp1_test_normal = os.path.join(cp1_test_path,'OK')
    cp1_test_abnormal = os.path.join(cp1_test_path,'NG')

    cp1_test_normal_files = sorted(glob.glob('{}/*'.format(cp1_test_normal)))
    for cp1_test_norfile in cp1_test_normal_files : 
        x_test_s_path.append(cp1_test_norfile)
        x_test_s.append(cp1_test_norfile)

    cp1_test_abnormal_files = sorted(glob.glob('{}/*'.format(cp1_test_abnormal)))
    for cp1_test_abnor in cp1_test_abnormal_files:
        x_test_b_path.append(cp1_test_abnor)
        x_test_b.append(cp1_test_abnor)
    
    #resize data
    X_train_s = resize_data(x_train_s)
    X_ref = resize_data(x_ref)
    y_ref = np.array(y_ref)

    X_test_s = resize_data(x_test_s)
    X_test_b = resize_data(x_test_b)
    # X_train_s : normal data
    # X_ref : reference data
    # X_test_s : test normal data
    # X_test_b : test abnormal data
    
    return X_train_s, X_ref, y_ref, X_test_s, X_test_b, x_test_s_path, x_test_b_path

def resize_data(path):
    x_out = []
    for item in path:
        img = cv2.imread(item)
        img = cv2.resize(img,(224,224))
        x_out.append(img.astype('float32') / 255)

    return np.array(x_out)
