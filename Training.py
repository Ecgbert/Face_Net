from Package.Data import Data, DataType, Draw
from Package.Model import *
from Package.Trainer import *



if __name__=='__main__':
    ## 資料
    #   1. 建立資料
    data = Data(filename='small_data', data_type=DataType.ALL_DATA)
    ## 模型
    #   1. 建立模型類別
    custom_model = CustomModel()
    #   2. 設定模型
    custom_model.set_model(load_model('D:\\Model\\Dense_FLN_LE.hdf5'))
    #   3. 遷移學習
    # custom_model.transfer_learning(custom_model.get_model())
    #   4. 設定模型的 loss function 與 最佳化方法
    custom_model.model_compile()
    #   5. 取得模型
    model = custom_model.get_model()
    # # 訓練器
    # #   1. 建立訓練器
    trainer = Trainer(data=data, model=model)
    #   2. 設定 callback function
    trainer.set_callbacks(filename='Small_LE')
    #   3. 開始訓練
    trainer.fit(filename='Small_LE')