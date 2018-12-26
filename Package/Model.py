from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, concatenate, Input, Dropout, Dense, Activation, MaxPooling2D, Conv2D, \
    AveragePooling2D, GlobalAveragePooling2D, regularizers, LSTM, TimeDistributed, merge, LeakyReLU
from keras import optimizers
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils, plot_model, get_custom_objects
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
import gc
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def swish(x):
    return K.sigmoid(x) * x


class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


get_custom_objects().update({"swish": Swish(swish)})


# 模型類別
class CustomModel(object):
    # 卷積層 + Batch Normalization Layer + Activate Function
    def conv2d_bn(self, x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None,
                  activate='relu'):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_initializer='he_normal',  # globor_uniform,
            kernel_regularizer = l2(0.0001),
            name = conv_name)(x)
        x = BatchNormalization(scale=True, name=bn_name)(x)
        if (activate != None):
            x = Activation(activate, name=name)(x)
        return x

    # 激活模型
    def model_compile(self, loss='categorical_crossentropy', opt='adamax', met=['acc']):
        self._model.compile(loss=loss, optimizer=opt, metrics=met)

    # 取得模型
    def get_model(self):
        return self._model

    # 設定模型
    def set_model(self, model):
        self._model = model

    # 遷移學習
    def transfer_learning(self, model):
        model.layers.pop()
        model.layers.pop()
        x = model.layers[-1].output
        x = Dense(6, activation='softmax', name='prob')(x)
        new_model = Model(inputs=model.input, outputs=x)
        self._model = new_model

# 提出之架構
class Dense_FaceLiveNet(CustomModel):
    def __init__(self, activate='swish', use_dense_block=True, use_global_average_pool=True):
        # 初始化
        #   Parameter
        #   1. activate                 : 激活函數
        #   2. use_dense_block          : 是否使用 Dense Block，若沒有使用，則為原本 FaceLiveNet 所使用之 Residual Block
        #   3. use_global_average_pool  : 是否使用 GlobalAveragePool，若沒有使用，則使用全連接層
        self.activate = activate
        self.use_dense_block = use_dense_block
        self.use_global_average_pool = use_global_average_pool
        self._model = self.build()
        self.model_compile()

    # 建立模型架構
    def build(self):
        get_custom_objects().update({"swish": Swish(swish)})
        # Backend 為 Tensorflow 定義 channel axis 為 3
        channel_axis = 3
        # 定義 Input 的大小
        input_shape = Input(shape=(224, 224, 1), name='data')

        # Stem layer
        net = self.conv2d_bn(input_shape, 32, 3, 3, strides=(2, 2), padding='valid', activate=self.activate)
        net = self.conv2d_bn(net, 32, 3, 3, strides=(1, 1), padding='valid', activate=self.activate)
        net = self.conv2d_bn(net, 64, 3, 3, strides=(1, 1), activate=self.activate)
        branch_0 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)
        net = self.conv2d_bn(branch_0, 80, 3, 3, strides=(2, 2), padding='valid', activate=self.activate)
        net = self.conv2d_bn(net, 192, 3, 3, strides=(1, 1), padding='valid', activate=self.activate)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(net)

        # inception1
        branch_0 = self.conv2d_bn(x, 96, 1, 1, strides=(1, 1), activate=self.activate)
        branch_1 = self.conv2d_bn(x, 64, 1, 1, strides=(1, 1), activate=self.activate)
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3, strides=(1, 1), activate=self.activate)
        branch_2 = self.conv2d_bn(x, 64, 1, 1, strides=(1, 1), activate=self.activate)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3, strides=(1, 1), activate=self.activate)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3, strides=(1, 1), activate=self.activate)
        x = [branch_0, branch_1, branch_2]
        mix1 = concatenate(x, axis=channel_axis)
        x = self.conv2d_bn(mix1, 96, 1, 1, strides=(1, 1), padding='valid', activate=self.activate)

        # inception2
        branch_0 = self.conv2d_bn(x, 64, 3, 3, strides=(1, 1), activate=self.activate)
        branch_1 = self.conv2d_bn(x, 96, 1, 1, strides=(1, 1), activate=self.activate)
        branch_1 = self.conv2d_bn(branch_1, 128, 3, 3, strides=(1, 1), activate=self.activate)
        branch_1 = self.conv2d_bn(branch_1, 160, 3, 3, strides=(1, 1), activate=self.activate)
        branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name="avg_pool_1")(x)
        if (self.use_dense_block):
            x1 = [x, branch_0, branch_1, branch_3]
        else:
            x1 = [branch_0, branch_1, branch_3]
        mix2 = concatenate(x1, axis=channel_axis)

        # inception3
        branch_0 = self.conv2d_bn(mix2, 192, 1, 1, strides=(1, 1), activate=self.activate)
        branch_1 = self.conv2d_bn(mix2, 128, 1, 1, strides=(1, 1), activate=self.activate)
        branch_1 = self.conv2d_bn(branch_1, 160, 1, 7, strides=(1, 1), activate=self.activate)
        branch_1 = self.conv2d_bn(branch_1, 160, 7, 1, strides=(1, 1), activate=self.activate)
        if (self.use_dense_block):
            x = [x, mix2, branch_0, branch_1]
        else:
            x = [mix2, branch_0, branch_1]
        mix3 = concatenate(x, axis=channel_axis, name='mixed3')

        # translate layer
        if (self.use_dense_block):
            x = self.conv2d_bn(mix3, 192, 1, 1, strides=(1, 1), padding='valid', activate=self.activate)
            x = AveragePooling2D((2, 2), strides=(2, 2))(x)
            x1 = BatchNormalization(scale=True, axis=channel_axis)(x)
        else:
            x1 = self.conv2d_bn(mix3, 192, 1, 1, strides=(1, 1), padding='valid', activate=self.activate)

        # inception4
        netb00 = self.conv2d_bn(x1, 192, 1, 1, strides=(1, 1), padding='same', activate=self.activate)
        netb10 = self.conv2d_bn(x1, 192, 1, 1, strides=(1, 1), padding='same', activate=self.activate)
        netb11 = self.conv2d_bn(netb10, 256, 3, 3, strides=(1, 1), padding='same', activate=self.activate)
        netb20 = self.conv2d_bn(x1, 160, 1, 1, strides=(1, 1), padding='same', activate=self.activate)
        netb21 = self.conv2d_bn(netb20, 192, 3, 3, strides=(1, 1), padding='same', activate=self.activate)
        netb22 = self.conv2d_bn(netb21, 256, 3, 3, strides=(1, 1), padding='same', activate=self.activate)
        netb30 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x1)
        netb31 = self.conv2d_bn(netb30, 160, 1, 1, strides=(1, 1), padding='same', activate=self.activate)
        if (self.use_dense_block):
            x = concatenate([x, netb00, netb11, netb22, netb31], axis=channel_axis, name='mixed4')
        else:
            x = concatenate([netb00, netb11, netb22, netb31], axis=channel_axis, name='mixed4')

        # inception5 * 2
        feature_list = [x]
        for _ in range(2):
            branch_0 = self.conv2d_bn(x, 256, 1, 1, strides=(1, 1), activate=self.activate)
            branch_1 = self.conv2d_bn(x, 128, 1, 3, strides=(1, 1), activate=self.activate)
            branch_1 = self.conv2d_bn(branch_1, 192, 3, 1, strides=(1, 1), activate=self.activate)
            branch_1 = self.conv2d_bn(branch_1, 256, 1, 3, strides=(1, 1), activate=self.activate)
            a = [branch_0, branch_1]
            mix5 = concatenate(a, axis=channel_axis)
            x1 = self.conv2d_bn(mix5, 256, 1, 1, strides=(1, 1), padding='valid', activate=self.activate)
            x = concatenate([x, x1], axis=channel_axis)
            feature_list.append(x)
        if (self.use_global_average_pool):
            x = concatenate(feature_list, axis=channel_axis)

        if (self.use_global_average_pool):
            # GlobalAveragePooling Layer
            x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        else:
            # Fully Connection Layer
            x = Dense(2000)(x)
            x = Dense(1000)(x)
        x = Dense(7, name='Logits')(x)
        x = Activation('softmax', name='prob')(x)
        model = Model(inputs=input_shape, outputs=x, name='FNet')
        return model
