"""
Exchange Rate Dataset Provider
"""
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_Exchange_Rate(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='exchange_rate.csv',
                 target='OT', scale=True, timeenc=0, freq='d', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 14  # 预测未来14天的汇率
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

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
        # 保存所有时间序列数据，用于短期预测任务
        self._prepare_timeseries()

    def _prepare_timeseries(self):
        """准备用于短期预测的时间序列数据"""
        # 为与M4兼容，只使用单列作为预测目标
        target_column = 0
        
        # 对于M4数据集，每个元素的timeseries是一个预测序列，长度为pred_len
        # 我们需要保持相同的格式
        
        # 先创建一个空列表
        self.timeseries = []
        self.ids = []
        
        # 在模型验证/测试时的行为：
        # 对每个数据样本，我们创建该样本接下来pred_len长度的预测目标
        if self.set_type > 0:  # 验证或测试集
            # 为了保持与M4一致，只创建单个样本
            # 这些样本将用于验证和测试阶段
            self.timeseries = [self.data_y[-self.pred_len:, target_column]]
            self.ids = ["TS_0"]
        else:  # 训练集
            # 为了在训练时充分利用数据，我们使用滑动窗口创建多个样本
            # 但这些样本主要用于计算训练损失，而非验证/测试
            # 为了兼容性，我们也只创建一个样本
            self.timeseries = [self.data_y[-self.pred_len:, target_column]]
            self.ids = ["TS_0"]
        
        # 注意：__getitem__方法仍然会使用全部训练数据，这是TimesNet的训练方式

    def last_insample_window(self):
        """
        返回最后的样本内窗口，用于短期预测
        需要与M4格式兼容，即返回[num_samples, seq_len]格式的数据
        """
        # 只有一个样本，与timeseries一致
        num_samples = len(self.timeseries)
        target_column = 0
        
        # 创建输入窗口，对所有样本都使用最后的seq_len个点
        insample = np.zeros((num_samples, self.seq_len))
        insample_mask = np.ones_like(insample)
        
        # 获取最后seq_len长度的历史数据
        last_window = self.data_x[-self.seq_len:, target_column]
        
        # 为所有样本填充相同的输入窗口
        for i in range(num_samples):
            insample[i, :] = last_window
        
        return insample, insample_mask

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # 确保数据有日期列，如果没有，则假设第一列是日期
        if 'date' not in df_raw.columns and 'Date' not in df_raw.columns:
            df_raw['date'] = pd.date_range(start='2000-01-01', periods=len(df_raw), freq=self.freq)
        else:
            date_col = 'date' if 'date' in df_raw.columns else 'Date'
            df_raw.rename(columns={date_col: 'date'}, inplace=True)
        
        # 转换日期为datetime格式
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        # 按日期排序
        df_raw = df_raw.sort_values(by='date')
        df_raw.reset_index(drop=True, inplace=True)
        
        # 拆分数据集
        # 默认使用70%数据训练，15%验证，15%测试
        train_ratio = 0.7
        valid_ratio = 0.15
        test_ratio = 0.15
        
        num_samples = len(df_raw)
        num_train = int(num_samples * train_ratio)
        num_valid = int(num_samples * valid_ratio)
        num_test = num_samples - num_train - num_valid
        
        border1s = [0, num_train - self.seq_len, num_train + num_valid - self.seq_len]
        border2s = [num_train, num_train + num_valid, num_samples]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 数据特征选择
        cols_data = df_raw.columns[1:]
        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[cols_data]
        elif self.features == 'S':
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
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
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