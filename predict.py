from keras.applications import MobileNetV2, VGG16, InceptionResNetV2
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, GlobalAveragePooling2D
from keras import backend as K
from keras.engine.network import Network
from keras.datasets import fashion_mnist
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import tensorflow as tf

from data import kyocera_data
from utils.evaluate import caculate_acc
import csv
import numpy as np
import matplotlib.pyplot as plt

input_shape = (224, 224, 3)

data_path = 'data'
X_train_s, X_ref, y_ref, X_test_s, X_test_b ,x_test_s_path, x_test_b_path = kyocera_data(data_path)

# mobile = MobileNetV2(include_top=True, input_shape=input_shape, alpha=alpha,
#                          depth_multiplier=1, weights='imagenet')
# mobile = InceptionResNetV2(include_top=False,  input_shape= input_shape, weights='imagenet')


mobile = VGG16(include_top=True, input_shape=input_shape, weights='imagenet')
mobile.layers.pop() 

model = Model(inputs=mobile.input,outputs=mobile.layers[-1].output)

# flat = GlobalAveragePooling2D()(mobile.layers[-1].output)
# model = Model(inputs=mobile.input,outputs=flat)

model.load_weights('model/model_t_smd_300.h5')

print(X_train_s.shape)
print(X_test_s.shape)
print(X_test_b.shape)

train = model.predict(X_train_s)
test_s = model.predict(X_test_s)
test_b = model.predict(X_test_b)

train = train.reshape((len(X_train_s),-1))
print('reshape train',train.shape)
test_s = test_s.reshape((len(X_test_s),-1))
print('reshape test normal',test_s.shape)
test_b = test_b.reshape((len(X_test_b),-1))
print('reshape test abnormal',test_b.shape)

print('fit model')
ms = MinMaxScaler()
train = ms.fit_transform(train)
test_s = ms.transform(test_s)
test_b = ms.transform(test_b)

# fit the model
clf = LocalOutlierFactor(n_neighbors=5)
y_pred = clf.fit(train)

# 異常スコア
Z1 = -clf._decision_function(test_s)
Z2 = -clf._decision_function(test_b)

#ROC曲線の描画
y_true = np.zeros(len(test_s)+len(test_b))
y_true[len(test_s):] = 1#0:正常、1：異常
path = x_test_s_path + x_test_b_path

# precision, recall, f1 = caculate_acc(y_true, np.hstack((Z1,Z2)),path)

fpr, tpr, _ = metrics.roc_curve(y_true, np.hstack((Z1, Z2)))

# AUC
auc = metrics.auc(fpr, tpr)
print('auc', auc)
