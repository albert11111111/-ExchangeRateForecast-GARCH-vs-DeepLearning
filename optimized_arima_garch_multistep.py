#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import time
from tqdm import tqdm
import pickle
import argparse
import sys

# 忽略警告
warnings.filterwarnings('ignore')

def load_and_prepare_data(root_path, data_path, target_col_name, log_return_col_name='log_return'):
    """
    加载数据，计算对数收益率，并为 ADF 检验和后续建模准备数据。
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
        print(f"警告: 目标列 '{target_col_name}' 包含零或负数值，可能导致对数收益率计算问题。正在移除这些行...")
        df = df[df[target_col_name] > 0].reset_index(drop=True)
        if df.empty:
            raise ValueError("移除零或负数值后，数据为空。")

    # 计算对数收益率
    df[log_return_col_name] = np.log(df[target_col_name] / df[target_col_name].shift(1))
    
    # 移除第一个NaN值（由于shift操作产生）
    df.dropna(subset=[log_return_col_name], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    if df.empty:
        raise ValueError("计算对数收益率并移除NaN后，数据为空。")
        
    return df

def perform_stationarity_test(series, series_name="序列"):
    """对给定序列执行ADF平稳性检验并打印结果。"""
    print(f"\n对 {series_name} 进行ADF平稳性检验:")
    # 确保序列中没有NaN值传递给adfuller
    adf_result = adfuller(series.dropna()) 
    print(f'ADF 统计量: {adf_result[0]}')
    print(f'p-值: {adf_result[1]}')
    print('临界值:')
    for key, value in adf_result[4].items():
        print(f'    {key}: {value}')
    if adf_result[1] <= 0.05:
        print(f"结论: p-值 ({adf_result[1]:.4f}) <= 0.05, 拒绝原假设，{series_name} 大概率是平稳的。")
    else:
        print(f"结论: p-值 ({adf_result[1]:.4f}) > 0.05, 不能拒绝原假设，{series_name} 大概率是非平稳的。")
    print("-" * 50)


def train_and_forecast_arima_garch(scaled_log_return_series_history, model_type, seq_len, pred_len, p=1, q=0, use_deterministic=True):
    """
    基于（标准化的）对数收益率历史训练ARIMA+GARCH模型并预测未来pred_len步的（标准化的）对数收益率。
    改进版支持AR和Constant均值模型，并针对不同GARCH族变体优化
    
    参数:
    - scaled_log_return_series_history: 标准化的对数收益率历史
    - model_type: 模型类型，例如'ARCH(1)_Const', 'GARCH(1,1)_AR'等
    - seq_len: 用于训练的序列长度
    - pred_len: 预测步数
    - p: ARIMA模型中AR项的阶数 (应用于AR均值模型)
    - q: ARIMA模型中MA项的阶数 (应用于AR均值模型)
    - use_deterministic: 是否使用确定性预测(True)或随机预测(False)
    
    返回:
    - 预测的标准化对数收益率，形状为(pred_len,)
    """
    # 去除确定性/随机标签，获取基本模型类型
    base_model_type = model_type
    if '_Det' in model_type or '_Rand' in model_type:
        base_model_type = model_type.split('_Det')[0].split('_Rand')[0]
    
    # 确定模型类型和均值设置
    is_ar_mean = '_AR' in base_model_type  # 使用AR均值
    is_const_mean = '_Const' in base_model_type  # 使用Constant均值
    
    # 使用序列的最后seq_len个点
    train_seq = scaled_log_return_series_history[-seq_len:]
    
    if is_ar_mean:
        # AR均值情况：先用ARIMA拟合得到均值预测和残差
        try:
            arima_model = ARIMA(train_seq, order=(p, 0, q))
            arima_results = arima_model.fit()
            arima_mean_forecast = arima_results.forecast(steps=pred_len)
            if hasattr(arima_mean_forecast, 'values'):
                arima_mean_forecast = arima_mean_forecast.values
            residuals = arima_results.resid
            
            # ARIMA参数 - 用于波动率预测后的再优化
            ar_coefs = arima_results.arparams if hasattr(arima_results, 'arparams') else []
            ma_coefs = arima_results.maparams if hasattr(arima_results, 'maparams') else []
            intercept = arima_results.params[0] if hasattr(arima_results, 'params') and len(arima_results.params) > 0 else 0
            
        except Exception as e_arima:
            print(f"警告: ARIMA({p},0,{q}) 模型拟合失败: {e_arima}。返回零预测。")
            return np.zeros(pred_len)
    elif is_const_mean:
        # Constant均值情况：直接使用序列均值作为常数预测值
        try:
            mean_value = np.mean(train_seq)
            arima_mean_forecast = np.ones(pred_len) * mean_value
            residuals = train_seq - mean_value
        except Exception as e_const:
            print(f"警告: 计算常数均值失败: {e_const}。返回零预测。")
            return np.zeros(pred_len)
    else:
        # 未指定有效的均值类型
        print(f"警告: 未指定有效的均值类型 (AR或Constant)，模型名称: {model_type}。返回零预测。")
        return np.zeros(pred_len)
    
    # 提取GARCH模型类型
    if 'ARCH(1)' in base_model_type:
        garch_type = 'ARCH'
        p_garch, o_garch, q_garch = 1, 0, 0
    elif 'GARCH(1,1)' in base_model_type:
        garch_type = 'GARCH'
        p_garch, o_garch, q_garch = 1, 0, 1
    elif 'EGARCH(1,1)' in base_model_type:
        garch_type = 'EGARCH'
        p_garch, o_garch, q_garch = 1, 1, 1
    elif 'GARCH-M(1,1)' in base_model_type:
        garch_type = 'GARCH-M'
        p_garch, o_garch, q_garch = 1, 0, 1
    else:
        print(f"警告: 未识别的GARCH模型类型: {base_model_type}。返回均值预测。")
        return arima_mean_forecast
    
    # 创建并拟合GARCH模型
    try:
        # 创建GARCH模型实例
        if garch_type == 'GARCH-M':
            # GARCH-M处理逻辑
            if is_const_mean:
                # 使用Constant均值的GARCH-M实现
                base_garch = arch_model(train_seq, mean='Constant', vol='GARCH', p=p_garch, q=q_garch)
                base_res = base_garch.fit(disp='off', show_warning=False)
                
                # 获取波动率和参数
                conditional_vol = base_res.conditional_volatility
                mu = base_res.params.get('mu', 0)
                omega = base_res.params.get('omega', 0)
                alpha = base_res.params.get('alpha[1]', 0)
                beta = base_res.params.get('beta[1]', 0)
                
                # 初始化预测
                last_resid = residuals[-1] if len(residuals) > 0 else 0
                last_var = conditional_vol[-1]**2 if len(conditional_vol) > 0 else np.var(residuals)
                
                # 风险溢价系数
                risk_premium = 0.15  # 使用更大的风险溢价系数
                
                # 多步预测
                forecasts = np.zeros(pred_len)
                forecast_vols = np.zeros(pred_len)
                
                for i in range(pred_len):
                    # 预测条件方差
                    sigma2 = omega + alpha * last_resid**2 + beta * last_var
                    sigma = np.sqrt(max(1e-12, sigma2))
                    forecast_vols[i] = sigma
                    
                    # GARCH-M均值 = 常数 + 风险溢价*波动率
                    mean_t = mu + risk_premium * sigma
                    
                    if not use_deterministic:
                        eps = np.random.normal(0, 1)
                        forecasts[i] = mean_t + sigma * eps
                        last_resid = sigma * eps
                    else:
                        forecasts[i] = mean_t
                        last_resid = 0
                    
                    last_var = sigma2
                
                return forecasts
            else:  # AR均值GARCH-M
                # 首先拟合波动率模型
                base_garch = arch_model(residuals, vol='GARCH', p=p_garch, q=q_garch, mean='Zero')
                base_res = base_garch.fit(disp='off', show_warning=False)
                
                # 获取波动率和参数
                conditional_vol_history = base_res.conditional_volatility
                omega = base_res.params.get('omega', 0)
                alpha = base_res.params.get('alpha[1]', 0)
                beta = base_res.params.get('beta[1]', 0)
                
                # 获取最后一个残差和条件方差
                last_resid = residuals[-1] if len(residuals) > 0 else 0
                last_var = conditional_vol_history[-1]**2 if len(conditional_vol_history) > 0 else np.var(residuals)
                
                # 预测多步波动率
                forecast_vols = np.zeros(pred_len)
                for i in range(pred_len):
                    sigma2 = omega + alpha * last_resid**2 + beta * last_var
                    sigma = np.sqrt(max(1e-12, sigma2))
                    forecast_vols[i] = sigma
                    
                    if not use_deterministic:
                        eps = np.random.normal(0, 1)
                        last_resid = sigma * eps
                    else:
                        last_resid = 0
                        
                    last_var = sigma2
                
                # 构建带有风险溢价的增强AR预测
                ar_mean_pred = arima_mean_forecast.copy()
                risk_premium = 0.15  # 使用更大的风险溢价系数
                
                for i in range(pred_len):
                    ar_mean_pred[i] += risk_premium * forecast_vols[i]
                
                if not use_deterministic:
                    for i in range(pred_len):
                        ar_mean_pred[i] += forecast_vols[i] * np.random.normal(0, 1)
                
                return ar_mean_pred
        else:  # 普通ARCH/GARCH/EGARCH模型
            # 确定均值模型和波动率模型
            mean_model = 'Zero'  # 拟合残差时使用零均值
            
            # 创建GARCH族模型
            if garch_type == 'ARCH':
                garch_model = arch_model(residuals, mean=mean_model, vol='ARCH', p=p_garch, q=0)
            elif garch_type == 'GARCH':
                garch_model = arch_model(residuals, mean=mean_model, vol='GARCH', p=p_garch, o=o_garch, q=q_garch)
            elif garch_type == 'EGARCH':
                try:
                    garch_model = arch_model(residuals, mean=mean_model, vol='EGARCH', p=p_garch, o=o_garch, q=q_garch, dist='t')
                except Exception as e_egarch_t:
                    print(f"警告: EGARCH dist='t' 创建失败: {e_egarch_t}。尝试正态分布。")
                    try:
                        garch_model = arch_model(residuals, mean=mean_model, vol='EGARCH', p=p_garch, o=o_garch, q=q_garch)
                    except Exception as e_egarch_norm:
                        print(f"警告: EGARCH正态分布也创建失败: {e_egarch_norm}。回退到GARCH。")
                        garch_model = arch_model(residuals, mean=mean_model, vol='GARCH', p=p_garch, q=q_garch)
            
            # 拟合GARCH模型
            garch_results = garch_model.fit(disp='off', options={'maxiter': 300})
            
            # 获取GARCH参数
            omega = garch_results.params.get('omega', 0)
            alpha_param = garch_results.params.get('alpha[1]', 0.0)
            beta_param = garch_results.params.get('beta[1]', 0.0)
            gamma_param = garch_results.params.get('gamma[1]', 0.0) if garch_type == 'EGARCH' else 0.0
            
            # 获取最后一个残差和条件方差
            last_resid = residuals[-1] if len(residuals) > 0 else 0
            last_var = garch_results.conditional_volatility[-1]**2 if len(garch_results.conditional_volatility) > 0 else np.var(residuals)
            
            # 多步预测
            forecasts_std_log_ret = np.zeros(pred_len)
            forecast_vols = np.zeros(pred_len)
            
            for i in range(pred_len):
                # 更新GARCH/EGARCH的条件方差预测
                if garch_type == 'EGARCH':
                    log_var = omega
                    abs_std_resid = abs(last_resid / np.sqrt(max(1e-12, last_var)))
                    log_var += alpha_param * (abs_std_resid - np.sqrt(2/np.pi))
                    log_var += gamma_param * (last_resid / np.sqrt(max(1e-12, last_var)))
                    sigma2 = np.exp(log_var)
                else:
                    sigma2 = omega + alpha_param * last_resid**2 + beta_param * last_var
                
                sigma = np.sqrt(max(1e-12, sigma2))
                forecast_vols[i] = sigma
                
                # 获取当前均值预测
                current_mean_pred = arima_mean_forecast[i]
                
                if not use_deterministic:
                    eps = np.random.normal(0, 1)
                    forecasts_std_log_ret[i] = current_mean_pred + sigma * eps
                    last_resid = sigma * eps
                else:
                    forecasts_std_log_ret[i] = current_mean_pred
                    last_resid = 0
                
                last_var = sigma2
            
            return forecasts_std_log_ret
    
    except Exception as e_garch:
        print(f"警告: {base_model_type} GARCH部分处理失败: {e_garch}。返回均值预测。")
        return arima_mean_forecast


def evaluate_model(y_true_prices, y_pred_prices):
    """
    评估模型性能(基于价格水平)
    """
    y_true_prices = np.asarray(y_true_prices).flatten()
    y_pred_prices = np.asarray(y_pred_prices).flatten()
    
    # 处理可能的NaN和Inf值
    if not np.all(np.isfinite(y_pred_prices)):
        finite_preds = y_pred_prices[np.isfinite(y_pred_prices)]
        mean_finite_pred = np.mean(finite_preds) if len(finite_preds) > 0 else (np.mean(y_true_prices) if len(y_true_prices) > 0 else 0)
        y_pred_prices = np.nan_to_num(y_pred_prices, nan=mean_finite_pred, posinf=1e12, neginf=-1e12)

    # 计算各种评估指标
    mse = mean_squared_error(y_true_prices, y_pred_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_prices, y_pred_prices)
    r2 = r2_score(y_true_prices, y_pred_prices)
    
    # 计算MAPE (避免除以零)
    mask = np.abs(y_true_prices) > 1e-9
    mape = np.mean(np.abs((y_true_prices[mask] - y_pred_prices[mask]) / y_true_prices[mask])) * 100 if np.sum(mask) > 0 else np.nan
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


def run_naive_baseline_forecast(data_df, original_price_col_name, pred_len, step_size, fixed_test_set_size=None):
    """
    运行朴素基线模型(使用前一天的价格作为所有未来步的预测)
    """
    original_prices = data_df[original_price_col_name].values
    num_total_points = len(original_prices)
    
    # 计算测试集起始索引
    if fixed_test_set_size is not None:
        # 使用固定大小的测试集
        test_start_idx = num_total_points - fixed_test_set_size
    else:
        # 使用70% 15% 15%的划分比例，与baseline一致
        test_start_idx = int(num_total_points * 0.85)  # 70% 训练 + 15% 验证
    
    if test_start_idx < 1:
        print(f"警告: 数据点总数 ({num_total_points}) 不足以进行有效预测。至少需要可以划分训练集。")
        return np.array([]), np.array([])
    
    # 确保循环不会超出界限
    end_loop_idx = num_total_points - pred_len
    
    # 收集实际值和预测值
    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []
    
    # 遍历测试集
    for i in tqdm(range(test_start_idx, end_loop_idx + 1, step_size), 
                 desc=f"Naive Baseline 预测进度 ({pred_len}天)", file=sys.stderr):
        if i < 1: continue  # 跳过第一个点(没有前一天的价格)
        
        # 获取本窗口的实际价格
        actual_price_levels_this_window = original_prices[i : i + pred_len]
        
        # 使用前一天的价格作为预测
        value_from_previous_day = original_prices[i-1]
        predicted_price_levels_this_window = [value_from_previous_day] * pred_len
        
        # 收集结果
        all_true_price_levels_collected.append(actual_price_levels_this_window)
        all_predicted_price_levels_collected.append(predicted_price_levels_this_window)
    
    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([])
    
    # 合并所有窗口的结果
    return np.concatenate(all_true_price_levels_collected), np.concatenate(all_predicted_price_levels_collected)


def rolling_forecast(data_df, original_price_col_name, log_return_col_name, 
                    model_type, seq_len, pred_len, step_size=1, p=1, q=0, fixed_test_set_size=None, use_deterministic=True):
    """
    使用滚动窗口执行多步预测
    
    参数:
    - data_df: 包含原始价格和对数收益率的DataFrame
    - original_price_col_name: 原始价格列名
    - log_return_col_name: 对数收益率列名
    - model_type: 模型类型字符串 (例如'GARCH(1,1)_Const')
    - seq_len: 用于训练的序列长度
    - pred_len: 预测步数
    - step_size: 滚动窗口步长
    - p: ARIMA模型的AR项阶数
    - q: ARIMA模型的MA项阶数
    - fixed_test_set_size: 固定测试集大小(如果为None则使用比例划分)
    - use_deterministic: 是否使用确定性预测模式
    
    返回:
    - 实际价格序列
    - 预测价格序列
    - None (兼容性参数)
    """
    # 提取对数收益率和原始价格
    log_returns = data_df[log_return_col_name].values
    original_prices = data_df[original_price_col_name].values
    
    # 标准化对数收益率
    log_return_scaler = StandardScaler()
    scaled_log_returns = log_return_scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
    
    # 计算测试集开始索引
    num_total_log_points = len(scaled_log_returns)
    
    if fixed_test_set_size is not None:
        # 使用固定大小的测试集
        test_start_idx = num_total_log_points - fixed_test_set_size
    else:
        # 使用70% 15% 15%的划分比例，与baseline一致
        test_start_idx = int(num_total_log_points * 0.85)  # 70% 训练 + 15% 验证
    
    # 检查数据是否足够
    min_history_needed = seq_len
    if test_start_idx < min_history_needed:
        print(f"警告: 模型 {model_type} 测试起始索引 ({test_start_idx}) < 最小所需历史长度 ({min_history_needed})。调整测试集起始点。")
        test_start_idx = min_history_needed  # 确保至少有seq_len的历史数据可用
    
    # 确保循环不会超出界限
    end_loop_idx = num_total_log_points - pred_len
    
    # 收集实际值和预测值
    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []
    
    # 设置极端值阈值(用于检测异常预测)
    extreme_value_threshold = 10.0
    
    # 滚动窗口预测
    for i in tqdm(range(test_start_idx, end_loop_idx + 1, step_size),
                 desc=f"{model_type} 预测进度 ({pred_len}天)", file=sys.stderr):
        
        # 获取历史数据(到当前点为止的所有数据)
        current_history_for_model_input = scaled_log_returns[:i]
        
        # 获取前一天的价格(用于重建价格水平)
        last_actual_price_for_reconstruction = original_prices[i-1]
        
        # 生成标准化对数收益率预测
        predicted_std_log_returns = train_and_forecast_arima_garch(
            current_history_for_model_input, model_type, seq_len, pred_len, p, q, use_deterministic=use_deterministic
        )
        
        # 检查预测的有效性
        use_fallback = False
        if not isinstance(predicted_std_log_returns, np.ndarray) or \
           predicted_std_log_returns.shape != (pred_len,):
            print(f"警告: {model_type} 返回了格式/形状错误的预测: {predicted_std_log_returns}")
            use_fallback = True
        elif np.any(np.isnan(predicted_std_log_returns)) or \
             np.any(np.isinf(predicted_std_log_returns)):
            print(f"警告: {model_type} 返回了NaN/Inf预测: {predicted_std_log_returns}")
            use_fallback = True
        elif np.any(np.abs(predicted_std_log_returns) > extreme_value_threshold):
            print(f"警告: {model_type} 返回了极端值预测 (abs > {extreme_value_threshold}): {predicted_std_log_returns}")
            use_fallback = True
        
        if use_fallback:
            # 使用零预测作为回退策略
            print(f"  使用零预测作为回退策略")
            predicted_std_log_returns = np.zeros(pred_len)
        
        # 将标准化对数收益率反变换为原始对数收益率
        predicted_log_returns_for_window = log_return_scaler.inverse_transform(
            np.array(predicted_std_log_returns).reshape(-1, 1)
        ).flatten()
        
        # 基于对数收益率重建价格水平
        reconstructed_price_forecasts_this_window = []
        current_reconstructed_price = last_actual_price_for_reconstruction
        
        for log_ret_pred_step in predicted_log_returns_for_window:
            # 限制极端值
            log_ret_pred_step = np.clip(log_ret_pred_step, -5, 5)
            
            # 计算下一步价格
            current_reconstructed_price = current_reconstructed_price * np.exp(log_ret_pred_step)
            reconstructed_price_forecasts_this_window.append(current_reconstructed_price)
        
        # 获取实际价格水平
        actual_price_levels_this_window = original_prices[i : i + pred_len]
        
        # 收集结果
        all_predicted_price_levels_collected.append(reconstructed_price_forecasts_this_window)
        all_true_price_levels_collected.append(actual_price_levels_this_window)
    
    # 检查是否有结果
    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([]), None
    
    # 合并所有窗口结果
    return np.concatenate(all_true_price_levels_collected), np.concatenate(all_predicted_price_levels_collected), None

def plot_results(results_dict, pred_len, results_dir_path, arima_params_label):
    """
    绘制预测结果图 (数据已经是原始价格水平)
    """
    sorted_model_keys = list(results_dict.keys())
    num_models_to_plot = len(sorted_model_keys)
    if num_models_to_plot == 0: 
        print("没有模型结果可供绘图。")
        return

    plt.figure(figsize=(15, 5 * num_models_to_plot))
    
    for plot_idx, model_type_key in enumerate(sorted_model_keys, 1):
        result_data = results_dict.get(model_type_key)
        if not (result_data and 'true_values' in result_data and 'predictions' in result_data and \
                len(result_data['true_values']) > 0 and len(result_data['predictions']) > 0):
            print(f"跳过绘图 {model_type_key}，缺少 true_values 或 predictions。")
            continue
            
        true_vals = np.asarray(result_data['true_values']).flatten()
        pred_vals = np.asarray(result_data['predictions']).flatten()
        
        plt.subplot(num_models_to_plot, 1, plot_idx)
        # 如果序列太长，只绘制一部分
        max_pts = min(1000, len(true_vals))
        plt.plot(true_vals[:max_pts], label='实际价格', color='blue')
        plt.plot(pred_vals[:max_pts], label='预测价格', color='red', linestyle='--')
        
        plt.title(f'{model_type_key} ({arima_params_label}) - 预测长度 {pred_len}天 (价格水平)')
        plt.xlabel('时间步长 (测试集样本)')
        plt.ylabel('汇率')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_path, f'plot_price_level_{arima_params_label}_pred_len_{pred_len}.png'))
    plt.close()


def generate_summary_table(all_results_summary, results_dir_path, arima_params_label):
    """
    生成结果汇总表 (基于价格水平的指标)
    """
    summary_data = {'Model': []}
    metric_names = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'Time(s)']
    
    # 获取所有预测长度
    pred_lengths_present = sorted(all_results_summary.keys())
    if not pred_lengths_present:
        print("没有结果可用于生成汇总表。")
        return pd.DataFrame()

    # 初始化列
    for pred_len_val in pred_lengths_present:
        for metric_n in metric_names:
            summary_data[f'{metric_n}_{pred_len_val}'] = []
    
    # 获取所有模型类型
    model_types_present_set = set()
    for pred_len_val in pred_lengths_present:
        if isinstance(all_results_summary[pred_len_val], dict):
            model_types_present_set.update(all_results_summary[pred_len_val].keys())
    
    # 排序模型列表
    ordered_model_types = sorted(list(model_types_present_set))
    
    if not ordered_model_types:
        print("在结果中未找到模型类型。")
        return pd.DataFrame()

    # 填充表格数据
    for model_t in ordered_model_types:
        summary_data['Model'].append(model_t)
        for pred_len_val in pred_lengths_present:
            model_result_data = all_results_summary.get(pred_len_val, {}).get(model_t)
            
            if model_result_data and 'metrics' in model_result_data and 'time' in model_result_data:
                for metric_n in metric_names:
                    if metric_n == 'Time(s)':
                        summary_data[f'{metric_n}_{pred_len_val}'].append(f"{model_result_data['time']:.2f}")
                    elif metric_n == 'MAPE':
                        value = model_result_data['metrics'].get(metric_n, float('nan'))
                        summary_data[f'{metric_n}_{pred_len_val}'].append(f"{value:.2f}%")
                    else:
                        value = model_result_data['metrics'].get(metric_n, float('nan'))
                        summary_data[f'{metric_n}_{pred_len_val}'].append(f"{value:.4f}")
            else:
                for metric_n in metric_names:
                    summary_data[f'{metric_n}_{pred_len_val}'].append('N/A')
    
    # 创建DataFrame并保存
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(results_dir_path, f'summary_table_{arima_params_label}.csv'), index=False)
    
    print(f"\n结果汇总 ({arima_params_label} - 固定测试集，多步预测):")
    print(df_summary.to_string())
    
    return df_summary 

def train_and_forecast_pure_arma(scaled_log_return_series_history, seq_len, pred_len, p_ar=1, q_ma=0):
    """
    基于标准化对数收益率历史训练纯ARMA模型并预测未来pred_len步的标准化对数收益率
    
    参数:
    - scaled_log_return_series_history: 标准化的对数收益率历史
    - seq_len: 用于训练的序列长度
    - pred_len: 预测步数
    - p_ar: ARMA模型AR项的阶数
    - q_ma: ARMA模型MA项的阶数
    
    返回:
    - 预测的标准化对数收益率，形状为(pred_len,)
    """
    train_seq = scaled_log_return_series_history[-seq_len:]
    
    arima_model = ARIMA(train_seq, order=(p_ar, 0, q_ma))
    try:
        arima_results = arima_model.fit()
        forecast_val = arima_results.forecast(steps=pred_len)
        if isinstance(forecast_val, pd.Series):  # 处理Series输出
            forecast_val = forecast_val.values
        # 确保返回一个与pred_len匹配的1D数组
        return np.array(forecast_val).flatten()[:pred_len]
    except Exception as e_arima:
        print(f"警告: ARMA({p_ar},0,{q_ma}) 模型拟合失败: {e_arima}。返回零预测。")
        return np.zeros(pred_len)

def rolling_forecast_pure_arma(data_df, original_price_col_name, log_return_col_name, 
                     seq_len, pred_len, step_size=1, p=1, q=0, fixed_test_set_size=None):
    """
    使用纯ARMA模型进行滚动预测
    
    参数:
    - data_df: 包含原始价格和对数收益率的DataFrame
    - original_price_col_name: 原始价格列名
    - log_return_col_name: 对数收益率列名
    - seq_len: 用于训练的序列长度
    - pred_len: 预测步数
    - step_size: 滚动窗口步长
    - p: ARMA模型的AR项阶数
    - q: ARMA模型的MA项阶数
    - fixed_test_set_size: 固定测试集大小(如果为None则使用比例划分)
    
    返回:
    - 实际价格序列
    - 预测价格序列
    - None (兼容性参数)
    """
    log_returns = data_df[log_return_col_name].values
    original_prices = data_df[original_price_col_name].values

    log_return_scaler = StandardScaler()
    scaled_log_returns = log_return_scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
    
    num_total_log_points = len(scaled_log_returns)
    
    # 计算测试集开始索引
    if fixed_test_set_size is not None:
        # 使用固定大小的测试集
        test_start_idx = num_total_log_points - fixed_test_set_size
    else:
        # 使用70% 15% 15%的划分比例，与baseline一致
        test_start_idx = int(num_total_log_points * 0.85)  # 70% 训练 + 15% 验证
    
    if test_start_idx < 0:
        print(f"警告: 纯ARMA模型计算的 test_start_idx ({test_start_idx}) < 0。没有足够的测试数据。")
        return np.array([]), np.array([]), None

    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []
    
    extreme_value_threshold = 10.0  # 与GARCH rolling forecast一致的阈值

    for i in tqdm(range(test_start_idx, num_total_log_points - pred_len + 1, step_size),
                 desc=f"Pure ARMA({p},{q}) 预测进度 ({pred_len}天)", file=sys.stderr):
        
        current_std_log_ret_history = scaled_log_returns[:i]
        last_actual_price_for_reconstruction = original_prices[i-1]

        predicted_std_log_returns = train_and_forecast_pure_arma(
            current_std_log_ret_history, seq_len, pred_len, p_ar=p, q_ma=q
        )
        
        # 对纯ARMA的预测也进行健全性检查
        if not (isinstance(predicted_std_log_returns, np.ndarray) and \
                predicted_std_log_returns.shape == (pred_len,) and \
                not np.any(np.isnan(predicted_std_log_returns)) and \
                not np.any(np.isinf(predicted_std_log_returns)) and \
                not np.any(np.abs(predicted_std_log_returns) > extreme_value_threshold)):
            print(f"警告: Pure ARMA({p},{q}) (在滚动窗口索引 {i}) 返回了无效/极端预测: {predicted_std_log_returns}. 使用零预测代替。")
            predicted_std_log_returns = np.zeros(pred_len)
        
        predicted_log_returns_for_window = log_return_scaler.inverse_transform(
            np.array(predicted_std_log_returns).reshape(-1, 1)
        ).flatten()

        reconstructed_price_forecasts_this_window = []
        current_reconstructed_price = last_actual_price_for_reconstruction
        for log_ret_pred_step in predicted_log_returns_for_window:
            log_ret_pred_step = np.clip(log_ret_pred_step, -5, 5) 
            current_reconstructed_price = current_reconstructed_price * np.exp(log_ret_pred_step)
            reconstructed_price_forecasts_this_window.append(current_reconstructed_price)
        
        actual_price_levels_this_window = original_prices[i : i + pred_len]
        
        all_predicted_price_levels_collected.append(reconstructed_price_forecasts_this_window)
        all_true_price_levels_collected.append(actual_price_levels_this_window)

    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([]), None 

    return np.concatenate(all_true_price_levels_collected), np.concatenate(all_predicted_price_levels_collected), None

def main(args):
    """
    主函数 - 执行模型训练、预测和评估
    """
    log_return_col = 'log_return_usd_jpy'
    fixed_test_set_size = None  # 设置为None，使用比例划分
    
    # 确定是否使用确定性预测模式
    use_deterministic = args.deterministic
    deterministic_label = "Det" if use_deterministic else "Rand"
    print(f"使用{'确定性' if use_deterministic else '随机'}预测模式")
    
    # 1. 加载并准备数据
    try:
        data_df = load_and_prepare_data(args.root_path, args.data_path, args.target, log_return_col)
    except ValueError as e:
        print(f"数据加载或预处理失败: {e}")
        return
    
    # 2. 检查数据大小是否足够
    num_total_points = len(data_df)
    
    # 使用70/15/15比例划分
    train_val_split = int(num_total_points * 0.7)
    val_test_split = int(num_total_points * 0.85)
    
    print(f"数据总点数: {num_total_points}")
    print(f"训练集大小: {train_val_split} (0-{train_val_split-1})")
    print(f"验证集大小: {val_test_split - train_val_split} ({train_val_split}-{val_test_split-1})")
    print(f"测试集大小: {num_total_points - val_test_split} ({val_test_split}-{num_total_points-1})")
    
    # 3. 检查训练集是否足够大
    if train_val_split < args.seq_len:
        print(f"错误: 训练数据点 ({train_val_split}) 少于模型序列长度 ({args.seq_len})。请减少 seq_len 或增加数据。")
        return
    
    # 4. 检查数据平稳性
    if not data_df[log_return_col].empty:
        perform_stationarity_test(data_df[log_return_col], series_name=f"{args.target} 的对数收益率 (USDJPY)")
    else:
        print("对数收益率序列为空。")
        return
    
    # 5. 定义模型和预测配置
    # 分离AR均值和Constant均值的模型，参考last150test.py的实现
    ar_mean_models = [
        'ARCH(1)_AR', 'GARCH(1,1)_AR', 'EGARCH(1,1)_AR', 'GARCH-M(1,1)_AR'
    ]
    constant_mean_models = [
        'ARCH(1)_Const', 'GARCH(1,1)_Const', 'EGARCH(1,1)_Const', 'GARCH-M(1,1)_Const'
    ]
    
    # 给模型名称添加确定性/随机标签
    if use_deterministic:
        ar_mean_models = [f"{m}_{deterministic_label}" for m in ar_mean_models]
        constant_mean_models = [f"{m}_{deterministic_label}" for m in constant_mean_models]
    
    # 预测步长
    pred_lens = [1, 7, 14, 30, 90, 180]
    
    # ARMA参数组合
    arma_params = [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    
    # 模型序列长度
    seq_lens_to_test = [args.seq_len]  # 只使用命令行参数指定的序列长度
    
    # 6. 初始化结果记录
    best_overall_metrics = {'MSE': float('inf'), 'model': None, 'p': None, 'q': None, 'seq_len': None, 'pred_len': None, 'metrics': None}
    all_configs_comparison = {}
    
    # 7. 遍历序列长度
    for current_seq_len in seq_lens_to_test:
        print(f"\n{'#'*80}")
        print(f"使用滚动窗口大小 (seq_len): {current_seq_len}")
        print(f"{'#'*80}")
        
        # 确保训练数据足够
        if train_val_split < current_seq_len:
            print(f"警告: 训练数据点 ({train_val_split}) 少于当前序列长度 ({current_seq_len})。跳过此序列长度。")
            continue
        
        # 8. 首先运行Constant均值模型组(不依赖ARMA参数)
        print(f"\n{'='*80}")
        print(f"运行Constant均值模型组 (seq_len={current_seq_len}, {deterministic_label}模式)")
        print(f"{'='*80}")
        
        # 创建结果目录 - 更改目录名以反映使用比例划分和确定性/随机模式
        constant_results_dir = f'arima_garch_results_logret_USDJPY_constant_seq{current_seq_len}_multistep_ratio_70_15_15_{deterministic_label}'
        os.makedirs(constant_results_dir, exist_ok=True)
        
        constant_run_results = {}
        constant_metrics = {}
        
        # 9. 先运行Naive Baseline
        print(f"\n{'-'*50}\n模型: Naive Baseline (PrevDay)\n{'-'*50}")
        
        for pred_len in pred_lens:
            start_time_naive = time.time()
            naive_actuals, naive_preds = run_naive_baseline_forecast(
                data_df, args.target, pred_len, args.step_size, fixed_test_set_size
            )
            elapsed_time_naive = time.time() - start_time_naive
            
            if len(naive_actuals) > 0:
                eval_metrics_naive = evaluate_model(naive_actuals, naive_preds)
                
                # 为每个预测长度创建唯一键
                naive_key = f"Naive_Baseline_step{pred_len}"
                
                constant_run_results[naive_key] = {
                    'metrics': eval_metrics_naive, 
                    'true_values': naive_actuals,
                    'predictions': naive_preds, 
                    'time': elapsed_time_naive
                }
                
                # 保存CSV结果
                num_rolling_windows_run = len(naive_actuals) // pred_len
                if num_rolling_windows_run > 0 and len(naive_actuals) % pred_len == 0:
                    # 重塑数据用于CSV
                    true_prices_csv = naive_actuals.reshape(num_rolling_windows_run, pred_len)
                    pred_prices_csv = naive_preds.reshape(num_rolling_windows_run, pred_len)
                    csv_cols = [f'pred_step_{j+1}' for j in range(pred_len)]
                    
                    df_true_prices_csv = pd.DataFrame(true_prices_csv, columns=csv_cols)
                    df_pred_prices_csv = pd.DataFrame(pred_prices_csv, columns=csv_cols)
                    
                    csv_true_fname = os.path.join(constant_results_dir, f'true_prices_Naive_Baseline_len{pred_len}.csv')
                    csv_pred_fname = os.path.join(constant_results_dir, f'pred_prices_Naive_Baseline_len{pred_len}.csv')
                    
                    df_true_prices_csv.to_csv(csv_true_fname, index=False)
                    df_pred_prices_csv.to_csv(csv_pred_fname, index=False)
                    print(f"已保存价格水平真实值CSV: {csv_true_fname}")
                    print(f"已保存价格水平预测值CSV: {csv_pred_fname}")
                
                constant_metrics[naive_key] = eval_metrics_naive
                print(f"预测步长 {pred_len}，执行时间: {elapsed_time_naive:.2f}秒")
                for name, val_metric in eval_metrics_naive.items():
                    print(f"{name}: {val_metric:.4f}{'%' if name == 'MAPE' else ''}")
                
                # 更新最佳结果
                if eval_metrics_naive['MSE'] < best_overall_metrics['MSE']:
                    best_overall_metrics.update({
                        'MSE': eval_metrics_naive['MSE'],
                        'model': naive_key,
                        'p': None,
                        'q': None,
                        'seq_len': current_seq_len,
                        'pred_len': pred_len,
                        'metrics': eval_metrics_naive
                    })
        
        # 10. 运行Constant均值模型
        for model_type in constant_mean_models:
            # 提取基本模型名称（去除确定性/随机标签）
            base_model_type = model_type.split('_Det')[0].split('_Rand')[0]
            
            print(f"\n{'='*50}")
            print(f"模型: {model_type}")
            print(f"{'='*50}")
            
            # 不同预测步长的测试
            for pred_len in pred_lens:
                print(f"\n{'-'*50}\n模型: {model_type} (预测步长={pred_len})\n{'-'*50}")
                
                start_time = time.time()
                
                # 执行模型预测
                actuals, preds, _ = rolling_forecast(
                    data_df, args.target, log_return_col, base_model_type,
                    current_seq_len, pred_len, args.step_size, p=1, q=0, fixed_test_set_size=fixed_test_set_size, use_deterministic=use_deterministic
                )
                
                elapsed_time = time.time() - start_time
                
                if len(actuals) > 0:
                    # 计算评估指标
                    eval_metrics = evaluate_model(actuals, preds)
                    
                    # 为这个模型+预测步长创建唯一键
                    key = f"{model_type}_step{pred_len}"
                    
                    # 保存结果
                    constant_run_results[key] = {
                        'metrics': eval_metrics,
                        'true_values': actuals,
                        'predictions': preds,
                        'time': elapsed_time
                    }
                    
                    # 保存CSV结果
                    model_name_clean = model_type.replace("(", "").replace(")", "").replace(",", "_").replace("-", "_")
                    num_rolling_windows_run = len(actuals) // pred_len
                    
                    if num_rolling_windows_run > 0 and len(actuals) % pred_len == 0:
                        # 重塑数据用于CSV
                        true_prices_csv = actuals.reshape(num_rolling_windows_run, pred_len)
                        pred_prices_csv = preds.reshape(num_rolling_windows_run, pred_len)
                        csv_cols = [f'pred_step_{j+1}' for j in range(pred_len)]
                        
                        df_true_prices_csv = pd.DataFrame(true_prices_csv, columns=csv_cols)
                        df_pred_prices_csv = pd.DataFrame(pred_prices_csv, columns=csv_cols)
                        
                        csv_true_fname = os.path.join(constant_results_dir, f'true_prices_{model_name_clean}_len{pred_len}.csv')
                        csv_pred_fname = os.path.join(constant_results_dir, f'pred_prices_{model_name_clean}_len{pred_len}.csv')
                        
                        df_true_prices_csv.to_csv(csv_true_fname, index=False)
                        df_pred_prices_csv.to_csv(csv_pred_fname, index=False)
                        print(f"已保存价格水平真实值CSV: {csv_true_fname}")
                        print(f"已保存价格水平预测值CSV: {csv_pred_fname}")
                    
                    constant_metrics[key] = eval_metrics
                    print(f"执行时间: {elapsed_time:.2f}秒")
                    for name, val_metric in eval_metrics.items():
                        print(f"{name}: {val_metric:.4f}{'%' if name == 'MAPE' else ''}")
                    
                    # 更新全局最佳结果
                    if eval_metrics['MSE'] < best_overall_metrics['MSE']:
                        best_overall_metrics.update({
                            'MSE': eval_metrics['MSE'],
                            'model': key,
                            'p': None,
                            'q': None,
                            'seq_len': current_seq_len,
                            'pred_len': pred_len,
                            'metrics': eval_metrics
                        })
        
        # 保存Constant均值模型结果
        if constant_run_results:
            # 按预测步长分组结果
            constant_results_by_pred_len = {}
            for key, value in constant_run_results.items():
                # 提取预测步长
                for pred_len in pred_lens:
                    if f"_step{pred_len}" in key:
                        if pred_len not in constant_results_by_pred_len:
                            constant_results_by_pred_len[pred_len] = {}
                        constant_results_by_pred_len[pred_len][key] = value
                        break
            
            # 绘制结果图表并生成汇总
            for pred_len, results in constant_results_by_pred_len.items():
                plot_results(results, pred_len, constant_results_dir, f"constant_seq{current_seq_len}_{deterministic_label}")
                generate_summary_table({pred_len: results}, constant_results_dir, f"constant_seq{current_seq_len}_step{pred_len}_{deterministic_label}")
            
            # 保存所有结果为pkl (可选)
            with open(os.path.join(constant_results_dir, f'all_results_USDJPY_constant_seq{current_seq_len}_multistep_{deterministic_label}.pkl'), 'wb') as f:
                pickle.dump(constant_run_results, f)
        
        # 11. 然后运行AR均值模型和ARMA (循环ARMA参数)
        for p_val, q_val in arma_params:
            print(f"\n{'='*80}")
            print(f"测试AR均值模型组 ARMA(p={p_val}, q={q_val}) (seq_len={current_seq_len}, {deterministic_label}模式)")
            print(f"{'='*80}")
            
            # 创建ARMA特定参数的目录
            arima_params_label_str = f"p{p_val}q{q_val}_seq{current_seq_len}_{deterministic_label}"
            ar_results_dir = f'arima_garch_results_logret_USDJPY_{arima_params_label_str}_multistep_ratio_70_15_15'
            os.makedirs(ar_results_dir, exist_ok=True)
            
            ar_run_results = {}
            ar_metrics = {}
            
            # 11.1 运行Pure ARMA
            for pred_len in pred_lens:
                arma_model_name_label = f'Pure_ARMA_{p_val}_{q_val}'
                print(f"\n{'-'*50}\n模型: {arma_model_name_label} (预测步长={pred_len})\n{'-'*50}")
                
                start_time_arma = time.time()
                arma_actuals, arma_preds, _ = rolling_forecast_pure_arma(
                    data_df, args.target, log_return_col,
                    current_seq_len, pred_len, args.step_size, p_val, q_val, fixed_test_set_size
                )
                elapsed_time_arma = time.time() - start_time_arma
                
                if len(arma_actuals) > 0:
                    eval_metrics_arma = evaluate_model(arma_actuals, arma_preds)
                    
                    # 为每个预测长度创建唯一键
                    arma_key = f"{arma_model_name_label}_step{pred_len}"
                    
                    ar_run_results[arma_key] = {
                        'metrics': eval_metrics_arma,
                        'true_values': arma_actuals,
                        'predictions': arma_preds,
                        'time': elapsed_time_arma
                    }
                    
                    # 保存CSV结果
                    num_rolling_windows_run = len(arma_actuals) // pred_len
                    if num_rolling_windows_run > 0 and len(arma_actuals) % pred_len == 0:
                        # 重塑数据用于CSV
                        true_prices_csv = arma_actuals.reshape(num_rolling_windows_run, pred_len)
                        pred_prices_csv = arma_preds.reshape(num_rolling_windows_run, pred_len)
                        csv_cols = [f'pred_step_{j+1}' for j in range(pred_len)]
                        
                        df_true_prices_csv = pd.DataFrame(true_prices_csv, columns=csv_cols)
                        df_pred_prices_csv = pd.DataFrame(pred_prices_csv, columns=csv_cols)
                        
                        csv_true_fname = os.path.join(ar_results_dir, f'true_prices_{arma_model_name_label}_len{pred_len}.csv')
                        csv_pred_fname = os.path.join(ar_results_dir, f'pred_prices_{arma_model_name_label}_len{pred_len}.csv')
                        
                        df_true_prices_csv.to_csv(csv_true_fname, index=False)
                        df_pred_prices_csv.to_csv(csv_pred_fname, index=False)
                        print(f"已保存价格水平真实值CSV: {csv_true_fname}")
                        print(f"已保存价格水平预测值CSV: {csv_pred_fname}")
                    
                    ar_metrics[arma_key] = eval_metrics_arma
                    print(f"预测步长 {pred_len}，执行时间: {elapsed_time_arma:.2f}秒")
                    for name, val_metric in eval_metrics_arma.items():
                        print(f"{name}: {val_metric:.4f}{'%' if name == 'MAPE' else ''}")
                    
                    # 更新最佳结果
                    if eval_metrics_arma['MSE'] < best_overall_metrics['MSE']:
                        best_overall_metrics.update({
                            'MSE': eval_metrics_arma['MSE'],
                            'model': arma_key,
                            'p': p_val,
                            'q': q_val,
                            'seq_len': current_seq_len,
                            'pred_len': pred_len,
                            'metrics': eval_metrics_arma
                        })
            
            # 11.2 运行AR均值的GARCH族模型
            for model_type in ar_mean_models:
                # 提取基本模型名称（去除确定性/随机标签）
                base_model_type = model_type.split('_Det')[0].split('_Rand')[0]
                
                print(f"\n{'='*50}")
                print(f"模型: {model_type} (AR p={p_val}, MA q={q_val})")
                print(f"{'='*50}")
                
                # 不同预测步长的测试
                for pred_len in pred_lens:
                    print(f"\n{'-'*50}\n模型: {model_type} (预测步长={pred_len})\n{'-'*50}")
                    
                    start_time = time.time()
                    
                    # 执行模型预测
                    actuals, preds, _ = rolling_forecast(
                        data_df, args.target, log_return_col, base_model_type,
                        current_seq_len, pred_len, args.step_size, p=p_val, q=q_val, fixed_test_set_size=fixed_test_set_size, use_deterministic=use_deterministic
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    if len(actuals) > 0:
                        # 计算评估指标
                        eval_metrics = evaluate_model(actuals, preds)
                        
                        # 为这个模型+预测步长创建唯一键
                        key = f"{model_type}_p{p_val}q{q_val}_step{pred_len}"
                        
                        # 保存结果
                        ar_run_results[key] = {
                            'metrics': eval_metrics,
                            'true_values': actuals,
                            'predictions': preds,
                            'time': elapsed_time
                        }
                        
                        # 保存CSV结果
                        model_name_clean = model_type.replace("(", "").replace(")", "").replace(",", "_").replace("-", "_")
                        num_rolling_windows_run = len(actuals) // pred_len
                        
                        if num_rolling_windows_run > 0 and len(actuals) % pred_len == 0:
                            # 重塑数据用于CSV
                            true_prices_csv = actuals.reshape(num_rolling_windows_run, pred_len)
                            pred_prices_csv = preds.reshape(num_rolling_windows_run, pred_len)
                            csv_cols = [f'pred_step_{j+1}' for j in range(pred_len)]
                            
                            df_true_prices_csv = pd.DataFrame(true_prices_csv, columns=csv_cols)
                            df_pred_prices_csv = pd.DataFrame(pred_prices_csv, columns=csv_cols)
                            
                            csv_true_fname = os.path.join(ar_results_dir, f'true_prices_{arima_params_label_str}_{model_name_clean}_len{pred_len}.csv')
                            csv_pred_fname = os.path.join(ar_results_dir, f'pred_prices_{arima_params_label_str}_{model_name_clean}_len{pred_len}.csv')
                            
                            df_true_prices_csv.to_csv(csv_true_fname, index=False)
                            df_pred_prices_csv.to_csv(csv_pred_fname, index=False)
                            print(f"已保存价格水平真实值CSV: {csv_true_fname}")
                            print(f"已保存价格水平预测值CSV: {csv_pred_fname}")
                        
                        ar_metrics[key] = eval_metrics
                        print(f"执行时间: {elapsed_time:.2f}秒")
                        for name, val_metric in eval_metrics.items():
                            print(f"{name}: {val_metric:.4f}{'%' if name == 'MAPE' else ''}")
                        
                        # 更新全局最佳结果
                        if eval_metrics['MSE'] < best_overall_metrics['MSE']:
                            best_overall_metrics.update({
                                'MSE': eval_metrics['MSE'],
                                'model': key,
                                'p': p_val,
                                'q': q_val,
                                'seq_len': current_seq_len,
                                'pred_len': pred_len,
                                'metrics': eval_metrics
                            })
            
            # 保存AR均值模型结果
            if ar_run_results:
                # 按预测步长分组结果
                ar_results_by_pred_len = {}
                for key, value in ar_run_results.items():
                    # 提取预测步长
                    for pred_len in pred_lens:
                        if f"_step{pred_len}" in key:
                            if pred_len not in ar_results_by_pred_len:
                                ar_results_by_pred_len[pred_len] = {}
                            ar_results_by_pred_len[pred_len][key] = value
                            break
                
                # 绘制结果图表并生成汇总
                for pred_len, results in ar_results_by_pred_len.items():
                    plot_results(results, pred_len, ar_results_dir, arima_params_label_str)
                    generate_summary_table({pred_len: results}, ar_results_dir, f"{arima_params_label_str}_step{pred_len}")
                
                # 保存所有结果为pkl (可选)
                with open(os.path.join(ar_results_dir, f'all_results_USDJPY_{arima_params_label_str}_multistep.pkl'), 'wb') as f:
                    pickle.dump(ar_run_results, f)
            
            # 更新配置比较结果
            all_configs_comparison[arima_params_label_str] = {
                'best_model_key': min(ar_metrics.keys(), key=lambda k: ar_metrics[k]['MSE']) if ar_metrics else None,
                'best_mse': min(r['MSE'] for r in ar_metrics.values()) if ar_metrics else float('inf'),
                'all_metrics': ar_metrics
            }
        
        # 将Constant均值模型的结果也添加到配置比较中
        constant_config_label = f"constant_seq{current_seq_len}_{deterministic_label}"
        if constant_metrics:
            all_configs_comparison[constant_config_label] = {
                'best_model_key': min(constant_metrics.keys(), key=lambda k: constant_metrics[k]['MSE']),
                'best_mse': min(r['MSE'] for r in constant_metrics.values()),
                'all_metrics': constant_metrics
            }
    
    # 13. 打印总结
    print("\n" + "="*80)
    print("所有配置的总结:")
    print("="*80)
    for config_label, summary_res in all_configs_comparison.items():
        print(f"\n配置: {config_label}")
        if summary_res['best_model_key']:
            print(f"  此配置中最佳模型: {summary_res['best_model_key']} (MSE: {summary_res['best_mse']:.4f})")
        else:
            print(f"  此配置中没有有效的模型结果")
    
    print("\n" + "="*80)
    print("全局最佳模型配置:")
    if best_overall_metrics['metrics']:
        print(f"模型类型: {best_overall_metrics['model']}")
        if best_overall_metrics['p'] is not None:
            print(f"ARMA参数: p={best_overall_metrics['p']}, q={best_overall_metrics['q']}")
        print(f"序列长度: {best_overall_metrics['seq_len']}")
        print(f"预测步长: {best_overall_metrics['pred_len']}")
        print(f"预测模式: {'确定性' if use_deterministic else '随机'}")
        print("\n评估指标:")
        for metric_name, value in best_overall_metrics['metrics'].items():
            print(f"  {metric_name}: {value:.4f}{'%' if metric_name == 'MAPE' else ''}")
    else:
        print("未能找到最佳模型。")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='优化版ARIMA+GARCH多步预测 (USDJPY, 对数收益率, 多步, 最后150点测试)')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='数据根目录')
    parser.add_argument('--data_path', type=str, default='英镑兑人民币_20250324_102930.csv', help='数据文件路径')
    parser.add_argument('--target', type=str, default='rate', help='原始汇率目标变量列名')
    parser.add_argument('--seq_len', type=int, default=96, help='ARIMA+GARCH历史对数收益率长度')
    parser.add_argument('--step_size', type=int, default=1, help='滚动窗口步长')
    parser.add_argument('--deterministic', action='store_true', help='使用确定性预测模式（不包含随机波动）')
    
    args = parser.parse_args()
    
    main(args) 