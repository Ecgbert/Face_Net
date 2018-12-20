from Package.Data import Data, DataType, Draw
from Package.Model import *
from Package.Trainer import *


if __name__ == '__main__':
    ## 畫出混淆矩陣
    #   1. 指定模型
    model = load_model('D:\\Train\\file.hdf5')
    #   2. 資料前處理
    x = Data.static_arr_reshape(arr=np.load('D:\\Dataset\\small_data_x.npy'))
    y = Data.static_one_hot_encoding(y=np.load('D:\\Dataset\\small_data_y.npy'))
    #   3. 呼叫方法畫出混淆矩陣
    Draw.confusion_matrix(model=model,
                          cls='le',
                          x=x,
                          y=y)
    #   4. 評估模型準確率
    Draw.evaluate(model=model, x=x, y=y)