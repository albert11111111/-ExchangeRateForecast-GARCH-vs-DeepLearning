#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import time
from tqdm import tqdm
import pickle
import argparse
import sys
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

# 设置中文字体
try:
    font = FontProperties(fname=r"C:\\Windows\\Fonts\\SimHei.ttf")
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"警告：无法加载中文字体 SimHei，图表中的中文可能显示为方框。错误: {e}")
    try:
        #尝试备用字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','FangSong'] # 尝试使用系统中的其他中文字体
        plt.rcParams['axes.unicode_minus'] = False
        print("已尝试设置备用中文字体。")
    except Exception as e_alt:
        print(f"警告: 备用中文字体也加载失败: {e_alt}")

# 忽略警告
warnings.filterwarnings('ignore')

# 设置PyTorch设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

def load_and_prepare_data(root_path, data_path, target_col_name, log_return_col_name='log_return'):
    """
    加载数据，计算对数收益率，并为后续建模准备数据。
    返回包含日期、原始价格和对数收益率的DataFrame。
    """
    df = pd.read_csv(os.path.join(root_path, data_path))
    
    date_col = 'date' if 'date' in df.columns else 'Date'
    df.rename(columns={date_col: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)

    if target_col_name not in df.columns:
        raise ValueError(f"目标列 '{target_col_name}' 在数据中未找到.")
    
    if not pd.api.types.is_numeric_dtype(df[target_col_name]):
        raise ValueError(f"目标列 '{target_col_name}' 必须是数值类型.")
    if (df[target_col_name] <= 0).any():
        original_len = len(df)
        df = df[df[target_col_name] > 0].reset_index(drop=True)
        print(f"警告: 目标列 '{target_col_name}' 包含零或负数值。移除了 {original_len - len(df)} 行。")
        if df.empty:
            raise ValueError("移除零或负数值后，数据为空。")

    # 计算对数收益率
    df[log_return_col_name] = np.log(df[target_col_name] / df[target_col_name].shift(1))
    df.dropna(subset=[log_return_col_name], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    if df.empty:
        raise ValueError("计算对数收益率并移除NaN后，数据为空。")
        
    return df

class FFNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(FFNNModel, self).__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU()) # 或 nn.Tanh()
            # layers.append(nn.Dropout(0.2)) # 可选的Dropout
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y).reshape(-1,1)

def train_and_predict_ffnn_step(scaled_log_return_history, lookback, pred_len, ffnn_config, device):
    """
    在每个滚动步骤中训练FFNN并进行预测。
    pred_len 在这里始终为1，因为我们进行一步预测然后重新训练。
    """
    model = FFNNModel(input_size=lookback, 
                      hidden_sizes=ffnn_config['hidden_sizes'], 
                      output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=ffnn_config['learning_rate'])

    X_train_all, y_train_all = create_sequences(scaled_log_return_history, lookback)

    if len(X_train_all) == 0:
        return np.zeros(pred_len)

    X_train_tensor = torch.FloatTensor(X_train_all).to(device)
    y_train_tensor = torch.FloatTensor(y_train_all).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=ffnn_config['batch_size'], shuffle=True, drop_last=True if len(X_train_tensor) > ffnn_config['batch_size'] else False)

    model.train()
    for epoch in range(ffnn_config['epochs']):
        epoch_loss = 0.0
        batch_count = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        # 计算并打印每个epoch的平均loss
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else epoch_loss
        print(f"    Epoch {epoch+1}/{ffnn_config['epochs']}, Loss: {avg_epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        last_sequence_np = scaled_log_return_history[-lookback:]
        if len(last_sequence_np) < lookback:
             return np.zeros(pred_len)
        last_sequence = torch.FloatTensor(last_sequence_np).reshape(1, lookback).to(device)
        prediction = model(last_sequence).cpu().numpy().flatten()
        
    return prediction[:pred_len]

def rolling_forecast_ffnn(data_df, original_price_col_name, log_return_col_name, 
                          lookback, pred_len, ffnn_config, fixed_test_set_size, device):
    log_returns = data_df[log_return_col_name].values
    original_prices = data_df[original_price_col_name].values

    num_total_log_points = len(log_returns)
    test_start_idx_log = num_total_log_points - fixed_test_set_size 
    
    if test_start_idx_log < lookback: 
        print(f"警告: FFNN 模型 test_start_idx_log ({test_start_idx_log}) 小于 lookback ({lookback})。没有足够的初始训练数据。")
        return np.array([]), np.array([])

    # 初始化模型（只初始化一次）
    model = FFNNModel(input_size=lookback, 
                      hidden_sizes=ffnn_config['hidden_sizes'], 
                      output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=ffnn_config['learning_rate'])

    # 初始训练（使用测试集之前的所有数据）
    initial_data = log_returns[:test_start_idx_log]
    scaler = StandardScaler()
    scaled_initial_data = scaler.fit_transform(initial_data.reshape(-1,1)).flatten()
    
    # 进行初始训练
    X_train_all, y_train_all = create_sequences(scaled_initial_data, lookback)
    X_train_tensor = torch.FloatTensor(X_train_all).to(device)
    y_train_tensor = torch.FloatTensor(y_train_all).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=ffnn_config['batch_size'], shuffle=True)

    print("进行初始训练...")
    for epoch in range(ffnn_config['epochs']):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else epoch_loss
        print(f"    初始训练 Epoch {epoch+1}/{ffnn_config['epochs']}, Loss: {avg_epoch_loss:.6f}")

    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []
    
    # 滚动预测
    progress_bar = tqdm(range(test_start_idx_log, num_total_log_points - pred_len + 1),
                        desc=f"FFNN(lb{config['lookback']},hs{'x'.join(map(str,config['hidden_sizes']))})", 
                        file=sys.stderr)
    
    for i in progress_bar:
        # 获取最新的序列用于预测
        current_sequence = log_returns[i-lookback:i]
        scaled_sequence = scaler.transform(current_sequence.reshape(-1,1)).flatten()
        
        # 预测
        model.eval()
        with torch.no_grad():
            X_pred = torch.FloatTensor(scaled_sequence).reshape(1, lookback).to(device)
            pred_scaled = model(X_pred).cpu().numpy().flatten()
            
        # 反向转换预测值
        predicted_log_return = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
        
        # 如果有新的真实值，进行增量更新
        if i < num_total_log_points - 1:
            model.train()
            # 准备新样本
            new_X = torch.FloatTensor(scaled_sequence).reshape(1, lookback).to(device)
            new_y = torch.FloatTensor(scaler.transform([[log_returns[i]]])).to(device)
            
            # 增量更新
            optimizer.zero_grad()
            output = model(new_X)
            loss = criterion(output, new_y)
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'update_loss': f'{loss.item():.6f}'})

        # 价格重构和收集结果
        last_actual_price = original_prices[i]
        predicted_price = last_actual_price * np.exp(predicted_log_return[0])
        
        actual_price = original_prices[i+1] if i+1 < len(original_prices) else None
        
        if actual_price is not None:
            all_predicted_price_levels_collected.append(predicted_price)
            all_true_price_levels_collected.append(actual_price)

    return np.array(all_true_price_levels_collected), np.array(all_predicted_price_levels_collected)


def evaluate_model(y_true_prices, y_pred_prices):
    y_true_prices = np.asarray(y_true_prices).flatten()
    y_pred_prices = np.asarray(y_pred_prices).flatten()
    
    if len(y_true_prices) == 0 or len(y_pred_prices) == 0 or len(y_true_prices) != len(y_pred_prices):
        #print(f"评估错误: 真实值或预测值为空或长度不匹配。真实值长度: {len(y_true_prices)}, 预测值长度: {len(y_pred_prices)}")
        return {'MSE': float('inf'), 'RMSE': float('inf'), 'MAE': float('inf'), 'MAPE': float('inf'), 'R2': float('-inf')}

    if not np.all(np.isfinite(y_pred_prices)):
        #尝试用真实值的均值填充，如果没有则用0
        num_nan_inf = np.sum(~np.isfinite(y_pred_prices))
        # print(f"警告: 预测值包含 {num_nan_inf} 个NaN/Inf值。尝试填充。")
        
        # 对于 MAPE，预测为0且真实值也为0的情况是允许的，但真实值为0预测非0会导致inf MAPE
        # 对于其他指标，用真实值的均值或者一个大的数来惩罚
        
        # 填充策略：
        # 1. 如果真实值存在且有限，用真实值的均值填充NaN/Inf
        # 2. 如果真实值不存在或也包含NaN/Inf，用一个较大的惩罚值或0填充
        if np.all(np.isfinite(y_true_prices)) and len(y_true_prices)>0:
            fill_value_for_nan = np.mean(y_true_prices)
            fill_value_for_posinf = np.max(y_true_prices) * 2 if np.max(y_true_prices) > 0 else 1e12 # 较大的惩罚
            fill_value_for_neginf = np.min(y_true_prices) * 2 if np.min(y_true_prices) < 0 else -1e12 # 较大的惩罚
        else: # 如果真实值也有问题，就用0或固定的大数
            fill_value_for_nan = 0
            fill_value_for_posinf = 1e12
            fill_value_for_neginf = -1e12

        y_pred_prices = np.nan_to_num(y_pred_prices, 
                                      nan=fill_value_for_nan, 
                                      posinf=fill_value_for_posinf, 
                                      neginf=fill_value_for_neginf)


    mse = mean_squared_error(y_true_prices, y_pred_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_prices, y_pred_prices)
    
    # R2 score can be negative if the model is arbitrarily worse.
    r2 = r2_score(y_true_prices, y_pred_prices)
    
    # MAPE calculation:
    # Mask out true values that are zero or very close to zero to avoid division by zero or extreme MAPE values.
    mask = np.abs(y_true_prices) > 1e-7 
    if np.sum(mask) == 0: # All true values are zero or near zero
        if np.allclose(y_pred_prices, 0): # Predictions are also zero
            mape = 0.0
        else: # Predictions are not zero, this is bad
            mape = float('inf') 
    else:
        mape = np.mean(np.abs((y_true_prices[mask] - y_pred_prices[mask]) / y_true_prices[mask])) * 100
    
    if not np.isfinite(mape): # Catch any other lingering issues with MAPE
        mape = float('inf')

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

def plot_error_analysis(true_vals, pred_vals, results_dir_path, ffnn_config_label, pred_len):
    """为特定模型绘制误差分析图"""
    if len(true_vals) == 0 or len(pred_vals) == 0 or len(true_vals) != len(pred_vals):
        print(f"错误分析图跳过: {ffnn_config_label} 数据不足或不匹配。")
        return

    errors = pred_vals - true_vals
    model_type_key = f"FFNN_{ffnn_config_label}"
    
    plt.figure(figsize=(20, 15))
    plt.suptitle(f'{model_type_key} - 预测误差分析 (预测长度 {pred_len}天)', fontsize=16)
    
    # 1. 误差分布图
    ax1 = plt.subplot(2, 2, 1)
    sns.histplot(errors, kde=True, bins=50, ax=ax1)
    ax1.set_title('预测误差分布', fontsize=12)
    ax1.set_xlabel('预测误差')
    ax1.set_ylabel('频数')
    
    if len(errors) >= 8: # stats.normaltest needs at least 8 samples in recent scipy versions
      try:
          stat_norm, p_value_norm = stats.normaltest(errors)
          ax1.text(0.05, 0.95, f'正态性检验 p值: {p_value_norm:.4f}', 
                   transform=ax1.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
      except ValueError as e:
          # print(f"正态性检验失败: {e}")
          ax1.text(0.05, 0.95, f'正态性检验失败 ({e})',
                  transform=ax1.transAxes, fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', alpha=0.5))
    else:
        ax1.text(0.05, 0.95, '样本过少无法进行正态性检验',
                  transform=ax1.transAxes, fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', fc='lightgrey', alpha=0.5))


    # 2. 实际值vs预测值散点图
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(true_vals, pred_vals, alpha=0.5, edgecolors='k', linewidths=0.5)
    min_val = min(true_vals.min(), pred_vals.min()) if len(true_vals)>0 and len(pred_vals)>0 and np.isfinite(true_vals).all() and np.isfinite(pred_vals).all() else 0
    max_val = max(true_vals.max(), pred_vals.max()) if len(true_vals)>0 and len(pred_vals)>0 and np.isfinite(true_vals).all() and np.isfinite(pred_vals).all() else 1
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想情况 (y=x)') # y=x line
    ax2.set_title('实际值 vs 预测值', fontsize=12)
    ax2.set_xlabel('实际值')
    ax2.set_ylabel('预测值')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # 3. 误差的自相关图 (ACF)
    ax3 = plt.subplot(2, 2, 3)
    if len(errors) > 20 : # plot_acf needs sufficient samples
        # Cap lags if series is short
        nlags = min(40, len(errors) // 2 - 1)
        if nlags > 0:
            plot_acf(errors, lags=nlags, alpha=0.05, ax=ax3, title='') # Removed title for suptitle
            ax3.set_title('预测误差的自相关函数 (ACF)', fontsize=12)
        else:
            ax3.text(0.5, 0.5, '样本过少无法绘制ACF', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, '样本过少无法绘制ACF', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
    ax3.set_xlabel('滞后阶数')
    ax3.set_ylabel('自相关系数')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # 4. 滚动RMSE (如果适用)
    ax4 = plt.subplot(2, 2, 4)
    window_size = max(1, min(20, len(true_vals) // 10)) # Dynamic window, at least 1
    if len(true_vals) > window_size and window_size > 0:
        rolling_rmse_values = []
        for k_roll in range(len(true_vals) - window_size +1):
             rolling_rmse_values.append(np.sqrt(mean_squared_error(true_vals[k_roll : k_roll+window_size], 
                                                                  pred_vals[k_roll : k_roll+window_size])))
        ax4.plot(rolling_rmse_values, label=f'滚动RMSE (窗口={window_size})')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, '样本过少无法绘制滚动RMSE', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
    ax4.set_title(f'滚动RMSE (窗口大小={window_size})', fontsize=12)
    ax4.set_xlabel('滚动窗口起始点')
    ax4.set_ylabel('RMSE')
    ax4.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    try:
        save_path = os.path.join(results_dir_path, f'error_analysis_{model_type_key}_pred_len_{pred_len}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"误差分析图已保存到: {save_path}")
    except Exception as e:
        print(f"保存误差分析图失败 for {model_type_key}: {e}")
    plt.close()

def plot_results(results_data, pred_len, results_dir_path, ffnn_config_label):
    true_vals = results_data.get('true_values')
    pred_vals = results_data.get('predictions')
    metrics = results_data.get('metrics')
    model_name_display = f"FFNN_{ffnn_config_label}"

    if true_vals is None or pred_vals is None or metrics is None or len(true_vals)==0 or len(pred_vals)==0:
        # print(f"绘图数据不完整或为空，跳过 {model_name_display} 的绘图。")
        return

    plt.figure(figsize=(18, 7))
    max_pts_plot = min(1000, len(true_vals)) # 限制绘图点数

    plt.plot(true_vals[:max_pts_plot], label='实际价格', color='#2E86C1', linewidth=1.5)
    plt.plot(pred_vals[:max_pts_plot], label='预测价格', color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.9)
    
    plot_title = f'{model_name_display} - 预测与实际价格 (预测长度 {pred_len}天, 显示前{max_pts_plot}点)'
    plt.title(plot_title, fontsize=14, pad=15)
    plt.xlabel('时间步长 (测试集样本)', fontsize=12)
    plt.ylabel('价格', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    metrics_text_display = (f"RMSE: {metrics.get('RMSE', float('nan')):.4f}\n"
                            f"MAE:  {metrics.get('MAE', float('nan')):.4f}\n"
                            f"MAPE: {metrics.get('MAPE', float('nan')):.2f}%\n"
                            f"R²:   {metrics.get('R2', float('nan')):.4f}")
    plt.text(0.02, 0.98, metrics_text_display, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.88, edgecolor='lightgrey', boxstyle='round,pad=0.5'),
             fontsize=9, verticalalignment='top', family='monospace', linespacing=1.5)
    
    plt.tight_layout(pad=1.0)
    try:
        save_path = os.path.join(results_dir_path, f'plot_price_level_{model_name_display}_pred_len_{pred_len}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"价格水平图已保存到: {save_path}")
    except Exception as e:
        print(f"保存价格水平图失败 for {model_name_display}: {e}")
    plt.close()

    # 调用误差分析绘图
    plot_error_analysis(true_vals, pred_vals, results_dir_path, ffnn_config_label, pred_len)


def generate_summary_table(all_ffnn_results, results_dir_path, base_filename_part):
    summary_data = {'Config_Label': [], 'Lookback':[], 'Hidden_Sizes':[], 'LR':[], 'Epochs':[], 'Batch_Size':[],
                    'MSE': [], 'RMSE': [], 'MAE': [], 'MAPE': [], 'R2': [], 'Time(s)': []}
    
    for config_key, data in all_ffnn_results.items():
        summary_data['Config_Label'].append(config_key)
        cfg = data.get('config', {})
        summary_data['Lookback'].append(cfg.get('lookback','N/A'))
        summary_data['Hidden_Sizes'].append(str(cfg.get('hidden_sizes','N/A')))
        summary_data['LR'].append(cfg.get('learning_rate','N/A'))
        summary_data['Epochs'].append(cfg.get('epochs','N/A'))
        summary_data['Batch_Size'].append(cfg.get('batch_size','N/A'))
        
        metrics = data.get('metrics', {})
        summary_data['MSE'].append(f"{metrics.get('MSE', float('nan')):.8f}")
        summary_data['RMSE'].append(f"{metrics.get('RMSE', float('nan')):.8f}")
        summary_data['MAE'].append(f"{metrics.get('MAE', float('nan')):.8f}")
        mape_val = metrics.get('MAPE', float('nan'))
        summary_data['MAPE'].append(f"{mape_val:.4f}%" if pd.notna(mape_val) and np.isfinite(mape_val) else ('inf%' if mape_val == float('inf') else 'nan'))
        summary_data['R2'].append(f"{metrics.get('R2', float('nan')):.8f}")
        summary_data['Time(s)'].append(f"{data.get('time', float('nan')):.2f}")

    df_summary = pd.DataFrame(summary_data)
    # Sort by RMSE (ascending, so NaNs/Infs will be at the bottom if not handled)
    # Convert RMSE to numeric for proper sorting, coercing errors for robust handling
    df_summary['RMSE_sort'] = pd.to_numeric(df_summary['RMSE'], errors='coerce')
    df_summary.sort_values(by='RMSE_sort', inplace=True, na_position='last')
    df_summary.drop(columns=['RMSE_sort'], inplace=True)

    
    try:
        table_path = os.path.join(results_dir_path, f'summary_table_FFNN_{base_filename_part}.csv')
        df_summary.to_csv(table_path, index=False)
        print(f"\nFFNN模型结果汇总 ({base_filename_part}) 已保存到 {table_path}:")
        # print(df_summary.to_string(index=False)) # Can be very long
    except Exception as e:
        print(f"保存汇总表失败: {e}")
    return df_summary


def main(args):
    log_return_col = f'log_return_{args.target_suffix}'
    fixed_test_set_size = args.test_size
    pred_len = 1 # 一步滚动预测

    try:
        data_df = load_and_prepare_data(args.root_path, args.data_path, args.target_col, log_return_col)
    except ValueError as e:
        print(f"数据加载或预处理失败: {e}")
        return

    if len(data_df) <= fixed_test_set_size + pred_len:
        print(f"错误: 数据点总数 ({len(data_df)}) 不足以分割出 {fixed_test_set_size} 点的测试集并进行 {pred_len} 步预测。")
        return
    
    num_total_points = len(data_df)
    num_train_points_for_first_window = num_total_points - fixed_test_set_size
    
    # 设置参数配置
    ffnn_configurations = []
    
    # 使用较大的学习率
    lookbacks = [120]  # 使用中等大小的lookback窗口
    hidden_layer_configs = [[16, 8]]  # 使用较宽的网络
    learning_rates = [0.1]  # 使用较大的学习率
    epochs_list = [100]  # 增加训练轮数
    batch_sizes = [64]  # 保持batch size不变

    for lb in lookbacks:
        if num_train_points_for_first_window < lb + 1:
            print(f"跳过 lookback={lb}, 因为初始训练数据 ({num_train_points_for_first_window}) 不足以创建至少一个训练样本 (需要 > {lb}).")
            continue
        for hs_config in hidden_layer_configs:
            for lr in learning_rates:
                for ep in epochs_list:
                    for bs in batch_sizes:
                        ffnn_configurations.append({
                            'lookback': lb,
                            'hidden_sizes': hs_config,
                            'learning_rate': lr,
                            'epochs': ep,
                            'batch_size': bs
                        })
    
    if not ffnn_configurations:
        print("错误: 没有有效的FFNN配置可测试。请检查lookback值、数据量和超参数解析。")
        return

    print(f"将测试 {len(ffnn_configurations)} 种FFNN配置...")

    all_ffnn_run_results = {}
    best_ffnn_metrics = {'RMSE': float('inf'), 'config_label': None, 'metrics': None, 'config_details': None}

    base_results_dir = f'ffnn_results_{args.target_suffix}_1step_last{fixed_test_set_size}test'
    os.makedirs(base_results_dir, exist_ok=True)

    global config
    for idx, config_item in enumerate(ffnn_configurations):
        config = config_item
        config_label = f"lb{config['lookback']}_hs{'x'.join(map(str,config['hidden_sizes']))}_lr{config['learning_rate']}_ep{config['epochs']}_bs{config['batch_size']}"
        print(f"\n{'='*80}")
        print(f"测试FFNN配置 {idx+1}/{len(ffnn_configurations)}: {config_label}")
        print(f"{'='*80}")

        current_config_results_dir = os.path.join(base_results_dir, config_label)
        os.makedirs(current_config_results_dir, exist_ok=True)

        start_time = time.time()
        true_prices, pred_prices = rolling_forecast_ffnn(
            data_df, args.target_col, log_return_col,
            config['lookback'], pred_len, config,
            fixed_test_set_size, DEVICE
        )
        elapsed_time = time.time() - start_time

        if len(true_prices) > 0 and len(pred_prices) > 0 and len(true_prices) == len(pred_prices):
            eval_metrics = evaluate_model(true_prices, pred_prices)
            print(f"执行时间: {elapsed_time:.2f}秒")
            for name, val_metric in eval_metrics.items():
                # 修改格式化字符串处理方式
                if isinstance(val_metric, (int, float)):
                    if name == 'MAPE':
                        print(f"  {name}: {val_metric:.2f}%")
                    else:
                        print(f"  {name}: {val_metric:.4f}")
                else:
                    print(f"  {name}: {val_metric}")

            current_run_data = {
                'metrics': eval_metrics, 
                'true_values': true_prices,
                'predictions': pred_prices, 
                'time': elapsed_time,
                'config': config 
            }
            all_ffnn_run_results[config_label] = current_run_data
            
            try:
                with open(os.path.join(current_config_results_dir, f'results_data_{config_label}.pkl'), 'wb') as f_pkl:
                    pickle.dump(current_run_data, f_pkl)
            except Exception as e_pkl:
                print(f"保存PKL文件失败 for {config_label}: {e_pkl}")

            plot_results(current_run_data, pred_len, current_config_results_dir, config_label)
            
            if pd.notna(eval_metrics['RMSE']) and eval_metrics['RMSE'] < best_ffnn_metrics['RMSE']:
                best_ffnn_metrics.update({
                    'RMSE': eval_metrics['RMSE'],
                    'config_label': config_label,
                    'metrics': eval_metrics,
                    'config_details': config
                })
        else:
            print(f"预测失败或返回结果长度不匹配 (True: {len(true_prices)}, Pred: {len(pred_prices)})，跳过此配置的评估和绘图。")
            all_ffnn_run_results[config_label] = { 
                'metrics': evaluate_model(np.array([]), np.array([])), 
                'true_values': np.array([]),
                'predictions': np.array([]),
                'time': elapsed_time,
                'config': config
            }

    if all_ffnn_run_results:
        summary_df = generate_summary_table(all_ffnn_run_results, base_results_dir, args.target_suffix)
        print("\n--- 配置评估结果 ---")
        print(summary_df[['Config_Label', 'RMSE', 'MAE', 'MAPE', 'R2', 'Time(s)']].to_string(index=False))

    print("\n" + "="*80)
    print("FFNN模型配置评估结果:")
    if best_ffnn_metrics['metrics'] and pd.notna(best_ffnn_metrics['RMSE']) and best_ffnn_metrics['RMSE'] != float('inf'):
        print(f"配置标签: {best_ffnn_metrics['config_label']}")
        print(f"配置参数: {best_ffnn_metrics['config_details']}")
        print("\n评估指标:")
        for metric_name, value in best_ffnn_metrics['metrics'].items():
            # 修改格式化字符串处理方式
            if isinstance(value, (int, float)):
                if metric_name == 'MAPE':
                    print(f"  {metric_name}: {value:.2f}%")
                else:
                    print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
    else:
        print("模型评估失败或RMSE为inf。")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FFNN汇率预测 (对数收益率, 单步, 最后N点测试)')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='数据根目录')
    parser.add_argument('--data_path', type=str, default='sorted_output_file.csv', help='数据文件路径 (例如 sorted_output_file.csv)')
    parser.add_argument('--target_col', type=str, default='rate', help='原始目标变量列名 (例如 rate)')
    parser.add_argument('--target_suffix', type=str, default='usd_jpy', help='目标系列的后缀，用于命名 (例如 usd_jpy)')
    parser.add_argument('--test_size', type=int, default=150, help='固定的测试集大小')
    
    # Hyperparameters for FFNN - can be lists for grid search
    parser.add_argument('--lookbacks', type=int, nargs='+', default=None, help='List of lookback window sizes (e.g., 20 40 60)')
    parser.add_argument('--hidden_layers', type=str, nargs='+', default=None, help='List of hidden layer configurations (e.g., "64,32" "128,64,32")')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=None, help='List of learning rates (e.g., 0.001 0.0005)')
    parser.add_argument('--epochs', type=int, nargs='+', default=None, help='List of epoch numbers (e.g., 50 100)')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=None, help='List of batch sizes (e.g., 32 64)')

    args = parser.parse_args()
    
    # Example of how hidden_layers might be parsed if provided as multiple strings for different layer structures
    # This is now handled inside main for simplicity with current arg structure
    # if args.hidden_layers:
    #     parsed_hidden_layers = []
    #     for hl_str_list in args.hidden_layers: # e.g. hl_str_list could be "64,32"
    #         parsed_hidden_layers.append([int(x.strip()) for x in hl_str_list.split(',')])
    #     args.hidden_layers = parsed_hidden_layers
            
    main(args) 