import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from enum import Enum
# 輸入資料型態
from sklearn.utils import shuffle

# 資料類別
class DataType(Enum):
    ALL_DATA = 'all data'       # 所有資料
    TRAIN = 'train data'        # 只有訓練資料
    TEST = 'test data'          # 只有測試資料
    CUSTOM = 'custom data'      # 原先設計好之資料
    SMALL = 'small dataset'     # 小型資料集

# 輸入資料
class Data(object):
    def __init__(self, data_type=DataType.ALL_DATA,
                 img_width=224, img_height=224,
                 split_data=False, filename='le', num_classes=6,
                 data_generator=True):
        print("- create data instance")
        try:
            print("\t- set data type ... ", end="")
            self.data_type=data_type
            self.num_classes=num_classes
            self.filename = filename
            print("{}".format(data_type.name))
        except:
            print("")
        try:
            print('\t- set X and Y ... ', end="")
            self.x = np.load('D:\\Research\\Dataset\\' + filename + '_x.npy')
            self.y = np.load('D:\\Research\\Dataset\\' + filename + '_y.npy')
            print("OK")
        except:
            print("")
        try:
            print("\t- set image shape ... ", end="")
            self.img_width=img_width
            self.img_height=img_height
            print("OK")
        except:
            print("image shape setting error")
        self.split_data=split_data
        try:
            print("\t- start data pre process ... ", end="")
            self.pre_process()
            self.data_generator = data_generator
            print("OK")
        except:
            print("error")


    # 資料預處理
    def pre_process(self, test_size=0.2, val_size=0.1, filename=None):
        x, y = shuffle(self.x, self.y)
        x = x.reshape(x.shape[0], self.img_width, self.img_height, 1)
        if (self.data_type == DataType.ALL_DATA):
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=25)
            train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=val_size, random_state=25)
        if (self.data_type == DataType.TRAIN):
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=25)
            val_x = test_x
            val_y = test_y
        if (self.data_type == DataType.TEST or self.data_type == DataType.SMALL):
            train_x = x
            train_y = y
            test_x = x
            test_y = y
            val_x = x
            val_y = y

        if (self.data_type == DataType.CUSTOM):
            self.train_x = self.arr_reshape(x)
            self.train_y = y
            self.test_x, self.test_y = shuffle(np.load('D:\\Dataset\\{}_test_x.npy'.format(filename)), np.load('Dataset\\{}_test_y.npy'.format(filename)))
            self.val_x, self.val_y = shuffle(np.load('D:\\Dataset\\{}_val_x.npy'.format(filename)), np.load('Dataset\\{}_val_y.npy'.format(filename)))
            self.test_x = self.arr_reshape(self.test_x)
            self.val_x = self.arr_reshape(self.val_x)

        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y
        self.one_hot_encoding()
        self.save_data()

    # 對輸出值進行 one hot encodcing
    def one_hot_encoding(self):
        try:
            pass
            self.train_y = np_utils.to_categorical(self.train_y, self.num_classes).astype('int')
            self.test_y = np_utils.to_categorical(self.test_y, self.num_classes).astype('int')
            self.val_y = np_utils.to_categorical(self.val_y, self.num_classes).astype('int')
        except:
            pass

    # 對 numpy 進行矩陣重組
    def arr_reshape(self, arr):
        return arr.reshape(arr.shape[0], self.img_width, self.img_height, 1)

    # 靜態取用矩陣重組
    @classmethod
    def static_arr_reshape(self, arr, img_width=224, img_height=224):
        return arr.reshape(arr.shape[0], img_width, img_height, 1)

    # 靜態取用 one hot encoding
    @classmethod
    def static_one_hot_encoding(self, y=None, classes=6):
        return np_utils.to_categorical(y, classes).astype('int')

    # 資料擴增
    def generator(self, batch_size):
        # train_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=10, zoom_range=0.2)
        # train_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=10)
        if self.data_generator:
            train_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=10, zoom_range=0.2)
            # train_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=10)
            train_datagen.fit(self.train_x)
            train_generator = train_datagen.flow(self.train_x, self.train_y, batch_size=batch_size)
            return train_generator
        else:
            train_datagen = ImageDataGenerator()
            train_datagen.fit(self.train_x)
            train_generator = train_datagen.flow(self.train_x, self.train_y, batch_size=batch_size)
            return train_generator

    # 資料儲存
    def save_data(self):
        np.save('D:\\Research\\Dataset\\' + self.filename + '_train_x.npy', self.train_x)
        np.save('D:\\Research\\Dataset\\' + self.filename + '_train_y.npy', self.train_y)
        np.save('D:\\Research\\Dataset\\' + self.filename + '_val_x.npy', self.val_x)
        np.save('D:\\Research\\Dataset\\' + self.filename + '_val_y.npy', self.val_y)
        np.save('D:\\Research\\Dataset\\' + self.filename + '_test_x.npy', self.test_x)
        np.save('D:\\Research\\Dataset\\' + self.filename + '_test_y.npy', self.test_y)

class Draw(object):
    # 評估模型準確率
    @classmethod
    def evaluate(self, model, x=None, y=None):
        loss, score = model.evaluate(x=x, y=y, batch_size=16, verbose=2)
        print("Accuracy：{:.2f}%".format(score * 100))

    # 混淆矩陣
    @classmethod
    def confusion_matrix(self, model=None, normalize=False, title='Confusion Matrix', cls='be', x=None, y=None):
        predict = model.predict(x=x, batch_size=16, verbose=2)
        pred = np.argmax(predict, axis=1)
        cnf_matrix = confusion_matrix(np.argmax(y, axis=1), pred)
        np.set_printoptions(precision=2)
        plt.figure()
        if cls == 'be':
            class_name = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        if cls == 'le':
            class_name = ["Frustration", "Confused", "Bored", "Delightful", "Flow", "Surprise"]
        Draw.__plot_confusion_matrix(cnf_matrix, classes=class_name,
                                title=title,
                                normalize=normalize)
        plt.show()

    @classmethod
    def __plot_confusion_matrix(self, cm, classes,
                                normalize=False,
                                title='Confusion matrix',
                                cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')