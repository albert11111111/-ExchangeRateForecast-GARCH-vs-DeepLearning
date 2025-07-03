"""
GBP/CNY Exchange Rate Dataset Provider
"""
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_GBP_CNY(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='英镑兑人民币_20250324_102930.csv',
                 target='rate', scale=True, timeenc=0, freq='d', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 30  # 预测未来30天的汇率
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        # seasonal_patterns参数不使用，但为了兼容性需要接收它
        self.seasonal_patterns = seasonal_patterns
        self.args = args

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # 确保数据有日期列
        if 'date' not in df_raw.columns and 'Date' not in df_raw.columns:
            raise ValueError("数据缺少日期列，请确保CSV文件包含'date'或'Date'列")
        
        # 统一日期列名
        date_col = 'date' if 'date' in df_raw.columns else 'Date'
        df_raw.rename(columns={date_col: 'date'}, inplace=True)
        
        # 转换日期为datetime格式
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        # 按日期排序
        df_raw = df_raw.sort_values(by='date')
        df_raw.reset_index(drop=True, inplace=True)
        
        # 拆分数据集
        # 使用70%数据训练，15%验证，15%测试
        train_ratio = 0.7
        valid_ratio = 0.15
        
        num_samples = len(df_raw)
        num_train = int(num_samples * train_ratio)
        num_valid = int(num_samples * valid_ratio)
        
        border1s = [0, num_train - self.seq_len, num_train + num_valid - self.seq_len]
        border2s = [num_train, num_train + num_valid, num_samples]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 数据特征选择
        if self.features == 'M' or self.features == 'MS':
            # 多变量情况
            df_data = df_raw.drop(['date'], axis=1)
        elif self.features == 'S':
            # 单变量情况
            df_data = df_raw[[self.target]]
        
        # 数据标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 时间特征
        df_stamp = df_raw[['date']][border1:border2]
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # 设置数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) 