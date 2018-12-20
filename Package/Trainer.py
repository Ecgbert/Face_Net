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
from Package.Data import Data, DataType

## 訓練器
class Trainer(object):
    def __init__(self, img_width=224, img_height=224, num_classes=7, epochs=50, batch_size=96, data=Data, model=None):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.set_model(model)
        self.data = data
    # 設定模型
    def set_model(self, model):
        self._model = model

    # 取得模型
    def get_model(self):
        return self._model

    # 訓練模型
    def fit(self, argument_amount=5, per_epoch_amount=220, filename="model", use_steps=False):
        if (use_steps):
            history = self._model.fit_generator(self.data.generator(batch_size=self.batch_size),
                                                epochs=self.epochs,
                                                #  steps_per_epoch=int(np.ceil(train_x.shape[0] / float(self.batch_size))),
                                                workers=4,
                                                # shuffle=False,
                                                samples_per_epoch=(len(self.data.train_x) * argument_amount),
                                                initial_epoch=0,
                                                validation_data=(self.data.val_x, self.data.val_y),
                                                # validation_steps=self.batch_size,
                                                callbacks=self.callbacks,
                                                verbose=1,
                                                # class_weight=self.class_weight
                                                )
        else:
            history = self._model.fit_generator(self.data.generator(batch_size=self.batch_size),
                                                epochs=self.epochs,
                                                steps_per_epoch=per_epoch_amount,
                                                #  steps_per_epoch=int(np.ceil(train_x.shape[0] / float(self.batch_size))),
                                                workers=4,
                                                # shuffle=False,
                                                initial_epoch=0,
                                                validation_data=(self.data.val_x, self.data.val_y),
                                                # validation_steps=self.batch_size,
                                                callbacks=self.callbacks,
                                                verbose=1,
                                                # class_weight=self.class_weight
                                                )
        pd.DataFrame(history.history).to_csv('{}_history.csv'.format(filename))
    # 建立檢查點與Early Stopping
    def set_callbacks(self, filename,
                      check_point=ModelCheckpoint("callbacks.{epoch:02d}-{val_acc:.5f}.hdf5", monitor='val_acc',
                                         verbose=1,
                                         save_best_only=True, mode='max'),
                      earlyStopping=EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=2, mode='max'),
                      reduce_lr=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1E-9)):
        origin_list = [check_point, earlyStopping, reduce_lr]
        callbacks = []
        for item in origin_list:
            if item is not None:
                callbacks.append(item)
        self.callbacks = callbacks