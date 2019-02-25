from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
# dataset
import os
import glob
def resize(x):
    x_out = []

    for i in range(len(x)):
        img = cv2.cvtColor(x[i], cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img,dsize=(96,96))
        x_out.append(img)

    return np.array(x_out)

def mnist_data(x_train,x_test,y_train,y_test):

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train_s, x_test_s, x_test_b = [], [], []
    x_ref, y_ref = [], []

    x_train_shape = x_train.shape

    for i in range(len(x_train)):
        if y_train[i] == 7:#スニーカーは7
            temp = x_train[i]
            x_train_s.append(temp.reshape((x_train_shape[1:])))
        else:
            temp = x_train[i]
            x_ref.append(temp.reshape((x_train_shape[1:])))
            y_ref.append(y_train[i])

    x_ref = np.array(x_ref)

    #refデータからランダムに6000個抽出
    number = np.random.choice(np.arange(0,x_ref.shape[0]),6000,replace=False)

    x, y = [], []

    x_ref_shape = x_ref.shape

    for i in number:
        temp = x_ref[i]
        x.append(temp.reshape((x_ref_shape[1:])))
        y.append(y_ref[i])

    x_train_s = np.array(x_train_s)
    x_ref = np.array(x)
    y_ref = to_categorical(y)

    #テストデータ
    for i in range(len(x_test)):
        if y_test[i] == 7:#スニーカーは7
            temp = x_test[i,:,:,:]
            x_test_s.append(temp.reshape((x_train_shape[1:])))

        if y_test[i] == 9:#ブーツは9
            temp = x_test[i,:,:,:]
            x_test_b.append(temp.reshape((x_train_shape[1:])))

    x_test_s = np.array(x_test_s)
    x_test_b = np.array(x_test_b)
    
    #resize image
    X_train_s = resize(x_train_s)
    X_ref = resize(x_ref)
    X_test_s = resize(x_test_s)
    X_test_b = resize(x_test_b)
    #X_train_s : normal data
    #X_ref : reference data
    # X_test_s : test normal data
    # X_test_b : test abnormal data
    return X_train_s, X_ref, y_ref, X_test_s, X_test_b

def kyocera_data(data_path):
    
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
        y_ref.append(np.array([1,0]))
    
    smd_train = os.path.join(smd_path, 'train', 'OK')
    smd_train_files = sorted(glob.glob('{}/*'.format(smd_train)))
    for smd_file in smd_train_files:
        x_ref.append(smd_file)
        y_ref.append(np.array([0,1]))

    # #make test data
    cp1_test_path =  os.path.join(cp1_path, 'test')
    # cp1_test_path =  os.path.join(smd_path, 'test')
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
        print(item)
        img = cv2.imread(item)
        img = cv2.resize(img,(96,96))
        x_out.append(img.astype('float32') / 255)

    return np.array(x_out)