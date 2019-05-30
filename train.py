from keras.applications import MobileNetV2, VGG16, InceptionResNetV2
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, GlobalAveragePooling2D
from keras import backend as K
from keras.engine.network import Network
from keras.datasets import fashion_mnist

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.callbacks import TensorBoard

from data import kyocera_data
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
input_shape = (139, 139, 3)
classes = 2
batchsize = 64
# feature_out = 512 #secondary network out for VGG16
# feature_out = 1280 #secondary network out for MobileNet
feature_out = 1536 # secondary network out for Inception Resnetv2
alpha = 0.5 #for MobileNetV2
lambda_ = 0.1 #for compact loss

def original_loss(y_true, y_pred):
    lc = 1/(classes*batchsize) * batchsize**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]   ) / ((batchsize-1)**2)
    return lc

def train(x_target, x_ref, y_ref, epoch_num):

    print("Model build...")
    # mobile = VGG16(include_top=False, input_shape=input_shape, weights='imagenet', pooling= 'avg')
    mobile = InceptionResNetV2(include_top=False,  input_shape= input_shape, weights='imagenet')
    # mobile = MobileNetV2(include_top=True, input_shape=input_shape, alpha=alpha,
    #                  weights='imagenet')

    mobile.layers.pop()
    '''
    Last layer :
    - block_13_expand : Mobilenet v2
    - block5_conv1  : VGG16
    - mixed_7a : Inception resnetv2
    '''
    # for layer in mobile.layers:
        # print(layer.name)
        # if layer.name == "block_13_expand": # "block5_conv1": for VGG16
        # if layer.name == "block5_conv1":
        # if layer.name == "mixed_7a":
        #     break
        # else:
        #     layer.trainable = False
    # exit()

    flat = GlobalAveragePooling2D()(mobile.layers[-1].output)
    model_t = Model(inputs=mobile.input,outputs=flat)

    model_r = Network(inputs=model_t.input,
                      outputs=model_t.output,
                      name="shared_layer")

    prediction = Dense(classes, activation='softmax')(model_t.output)
    model_r = Model(inputs=model_r.input,outputs=prediction)

    optimizer = SGD(lr=5e-5, decay=0.00005)
    model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
    model_t.compile(optimizer=optimizer, loss=original_loss)

    model_t.summary()

    ref_samples = np.arange(x_ref.shape[0])
    loss, loss_c = [], []
    epochs = []
    print("training...")

    for epochnumber in range(epoch_num):
        x_r, y_r, lc, ld = [], [], [], []

        np.random.shuffle(x_target)

        np.random.shuffle(ref_samples)
        for i in range(len(x_target)):
            x_r.append(x_ref[ref_samples[i]])
            y_r.append(y_ref[ref_samples[i]])
        x_r = np.array(x_r)
        y_r = np.array(y_r)

        for i in range(int(len(x_target) / batchsize)):
            batch_target = x_target[i*batchsize:i*batchsize+batchsize]
            batch_ref = x_r[i*batchsize:i*batchsize+batchsize]
            batch_y = y_r[i*batchsize:i*batchsize+batchsize]
            #target data
            lc.append(model_t.train_on_batch(batch_target, np.zeros((batchsize, feature_out))))

            #reference data
            ld.append(model_r.train_on_batch(batch_ref,batch_y))

        loss.append(np.mean(ld))
        loss_c.append(np.mean(lc))
        epochs.append(epochnumber)

        print("epoch : {} ,Descriptive loss : {}, Compact loss : {}".format(epochnumber+1, loss[-1], loss_c[-1]))
        if epochnumber % 10 == 0:
            model_t.save_weights('model/model_t_smd_{}.h5'.format(epochnumber))
            model_r.save_weights('model/model_r_smd_{}.h5'.format(epochnumber))

if __name__ == "__main__":
    #data_path = 'D:\Project\deep_one\Deep_Descriptive\data'
    data_path = '/home/asilla/hanh/Deep_Descriptive/data/kyocera' 
    X_train_s, X_ref, y_ref, X_test_s, X_test_b, _, _ = kyocera_data(data_path)
    train(X_train_s, X_ref, y_ref, 600)

    
