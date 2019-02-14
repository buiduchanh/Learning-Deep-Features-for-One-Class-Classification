from keras.applications import MobileNetV2, VGG16
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras import backend as K
from keras.engine.network import Network
from keras.datasets import fashion_mnist

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

from data import makedata

input_shape = (96, 96, 3)
classes = 10
batchsize = 128
feature_out = 512 #secondary network out for VGG16
# feature_out = 1280 #secondary network out for MobileNet
alpha = 0.5 #for MobileNetV2
lambda_ = 0.1 #for compact loss

def original_loss(y_true, y_pred):
    lc = 1/(classes*batchsize) * batchsize**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]) / ((batchsize-1)**2)
    return lc

def train(x_target, x_ref, y_ref, epoch_num):

    # VGG16読み込み, S network用
    print("Model build...")
    mobile = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')

    # mobile net読み込み, S network用
    # mobile = MobileNetV2(include_top=True, input_shape=input_shape, alpha=alpha,
    #                      depth_multiplier=1, weights='imagenet')

    #最終層削除
    mobile.layers.pop() 

    # 重みを固定
    for layer in mobile.layers:
        # if layer.name == "block_13_expand": # "block5_conv1": for VGG16
        if layer.name == "block5_conv1":
            break
        else:
            layer.trainable = False

    model_t = Model(inputs=mobile.input,outputs=mobile.layers[-1].output)

    # R network用　Sと重み共有
    model_r = Network(inputs=model_t.input,
                      outputs=model_t.output,
                      name="shared_layer")

    #Rに全結合層を付ける
    prediction = Dense(classes, activation='softmax')(model_t.output)
    model_r = Model(inputs=model_r.input,outputs=prediction)

    #コンパイル
    optimizer = SGD(lr=5e-5, decay=0.00005)
    model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
    model_t.compile(optimizer=optimizer, loss=original_loss)

    # model_t.summary()
    # model_r.summary()

    print("x_target is",x_target.shape[0],'samples')
    print("x_ref is",x_ref.shape[0],'samples')

    ref_samples = np.arange(x_ref.shape[0])
    loss, loss_c = [], []

    print("training...")

    #学習
    for epochnumber in range(epoch_num):
        x_r, y_r, lc, ld = [], [], [], []

        #ターゲットデータシャッフル
        np.random.shuffle(x_target)

        #リファレンスデータシャッフル
        np.random.shuffle(ref_samples)
        for i in range(len(x_target)):
            x_r.append(x_ref[ref_samples[i]])
            y_r.append(y_ref[ref_samples[i]])
        x_r = np.array(x_r)
        y_r = np.array(y_r)

        for i in range(int(len(x_target) / batchsize)):

            #batchsize分のデータロード
            batch_target = x_target[i*batchsize:i*batchsize+batchsize]
            batch_ref = x_r[i*batchsize:i*batchsize+batchsize]
            batch_y = y_r[i*batchsize:i*batchsize+batchsize]

            #target data
            #学習しながら、損失を取得
            lc.append(model_t.train_on_batch(batch_target, np.zeros((batchsize, feature_out))))

            #reference data
            #学習しながら、損失を取得
            ld.append(model_r.train_on_batch(batch_ref, batch_y))

        loss.append(np.mean(ld))
        loss_c.append(np.mean(lc))

        print("epoch : {} ,Descriptive loss : {}, Compact loss : {}".format(epochnumber+1, loss[-1], loss_c[-1]))
        model_t.save_weights('model/model_t_{}.h5'.format(epochnumber))



if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    X_train_s, X_ref, y_ref, X_test_s, X_test_b = makedata(x_train,x_test,y_train,y_test)
    train(X_train_s, X_ref, y_ref, 500)

    