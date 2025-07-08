#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 从主脚本导入数据处理函数
from run_arima_garch_jpy_last150test import load_and_prepare_data

# 简化版测试，直接对比EGARCH和ARMA
def main():
    print("开始EGARCH与ARMA对比测试...")
    
    # 加载数据
    root_path = './dataset/'
    data_path = 'sorted_output_file.csv'
    target = 'rate'
    log_return_col = 'log_return_usd_jpy'
    
    try:
        print("加载数据...")
        data_df = load_and_prepare_data(root_path, data_path, target, log_return_col)
        
        # 只使用较小的数据集进行测试
        data_df = data_df.tail(200).reset_index(drop=True)
        print(f"成功加载数据，样本数量: {len(data_df)}")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 简化的测试
    try:
        # 获取对数收益率
        log_returns = data_df[log_return_col].values
        
        # 标准化
        log_return_scaler = StandardScaler()
        scaled_log_returns = log_return_scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
        
        # 使用简单的训练-测试分割
        train_size = 180
        train_data = scaled_log_returns[:train_size]
        test_data = scaled_log_returns[train_size:]
        
        print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
        
        # 拟合ARMA模型
        print("\n拟合ARMA(1,0)模型...")
        arima_model = ARIMA(train_data, order=(1, 0, 0))
        arima_results = arima_model.fit()
        print("ARMA模型拟合完成.")
        
        # 拟合EGARCH模型
        print("\n拟合EGARCH(1,1)模型...")
        egarch_model = arch_model(train_data, mean='AR', vol='EGARCH', p=1, q=1, o=1, lags=1)
        egarch_results = egarch_model.fit(disp='off', show_warning=False)
        print("EGARCH模型拟合完成.")
        
        # 进行单步预测
        test_len = len(test_data)
        arma_preds = arima_results.forecast(steps=test_len)
        if hasattr(arma_preds, 'values'):
            arma_preds = arma_preds.values
            
        # 对EGARCH进行逐步预测
        egarch_preds = np.zeros(test_len)
        
        print("\n开始进行EGARCH逐步预测...")
        temp_data = train_data.copy()
        for i in range(test_len):
            print(f"预测步骤 {i+1}/{test_len}...")
            forecast = egarch_results.forecast(horizon=1)
            next_step_pred = forecast.mean.values[-1, 0]
            egarch_preds[i] = next_step_pred
            
            # 更新数据集
            temp_data = np.append(temp_data, test_data[i])
            if (i+1) % 5 == 0:  # 每5步重新拟合模型
                print(f"在步骤 {i+1} 重新拟合EGARCH模型...")
                new_model = arch_model(temp_data, mean='AR', vol='EGARCH', p=1, q=1, o=1, lags=1)
                egarch_results = new_model.fit(disp='off', show_warning=False)
        
        # 反标准化
        arma_preds_unscaled = log_return_scaler.inverse_transform(arma_preds.reshape(-1, 1)).flatten()
        egarch_preds_unscaled = log_return_scaler.inverse_transform(egarch_preds.reshape(-1, 1)).flatten()
        
        # 比较预测结果
        print("\n预测结果对比:")
        for i in range(min(5, len(arma_preds), len(egarch_preds))):
            print(f"步骤 {i+1}: ARMA = {arma_preds_unscaled[i]:.6f}, EGARCH = {egarch_preds_unscaled[i]:.6f}")
            
        # 计算MSE
        arma_mse = mean_squared_error(log_returns[train_size:train_size+len(arma_preds)], arma_preds_unscaled)
        egarch_mse = mean_squared_error(log_returns[train_size:train_size+len(egarch_preds)], egarch_preds_unscaled)
        
        print(f"\nARMA MSE: {arma_mse:.8f}")
        print(f"EGARCH MSE: {egarch_mse:.8f}")
        
        if arma_mse > egarch_mse:
            improvement = (arma_mse - egarch_mse) / arma_mse * 100
            print(f"EGARCH比ARMA改进了{improvement:.2f}%")
        else:
            degradation = (egarch_mse - arma_mse) / arma_mse * 100
            print(f"EGARCH比ARMA差了{degradation:.2f}%")
            
        # 比较异同
        print(f"\nEGARCH和ARMA预测值的相关系数: {np.corrcoef(egarch_preds, arma_preds)[0,1]:.4f}")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 