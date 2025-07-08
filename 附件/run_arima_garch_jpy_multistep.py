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
import matplotlib as mpl

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_and_prepare_data(root_path, data_path, target_col_name, log_return_col_name='log_return'):
    """
    加载数据，计算对数收益率，并为ADF检验和后续建模准备数据。
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
    df.reset_index(drop=True, inplace=True) #确保索引重置
    
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

def evaluate_model(y_true_prices, y_pred_prices):
    """计算各种评估指标，包括MSE、RMSE、MAE、MAPE和R2。"""
    y_true_prices = np.asarray(y_true_prices).flatten()
    y_pred_prices = np.asarray(y_pred_prices).flatten()
    
    if not np.all(np.isfinite(y_pred_prices)):
        finite_preds = y_pred_prices[np.isfinite(y_pred_prices)]
        mean_finite_pred = np.mean(finite_preds) if len(finite_preds) > 0 else (np.mean(y_true_prices) if len(y_true_prices) > 0 else 0)
        y_pred_prices = np.nan_to_num(y_pred_prices, nan=mean_finite_pred, posinf=1e12, neginf=-1e12)

    mse = mean_squared_error(y_true_prices, y_pred_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_prices, y_pred_prices)
    r2 = r2_score(y_true_prices, y_pred_prices)
    
    mask = np.abs(y_true_prices) > 1e-9
    mape = np.mean(np.abs((y_true_prices[mask] - y_pred_prices[mask]) / y_true_prices[mask])) * 100 if np.sum(mask) > 0 else np.nan
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

def split_data(data_df, train_ratio=0.7, val_ratio=0.15):
    """
    按照指定比例划分训练集、验证集和测试集
    """
    n = len(data_df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_df = data_df.iloc[:train_size]
    val_df = data_df.iloc[train_size:train_size+val_size]
    test_df = data_df.iloc[train_size+val_size:]
    
    print(f"数据集总大小: {n}")
    print(f"训练集大小: {len(train_df)} ({len(train_df)/n:.2%})")
    print(f"验证集大小: {len(val_df)} ({len(val_df)/n:.2%})")
    print(f"测试集大小: {len(test_df)} ({len(test_df)/n:.2%})")
    
    return train_df, val_df, test_df

def train_and_forecast_arima_garch_multistep(scaled_log_return_series_history, model_type, seq_len, pred_len, p_ar=1, q_ma=0):
    """
    训练ARIMA-GARCH模型并进行多步预测
    
    使用arch库的forecast功能进行多步预测，针对不同模型使用适当的预测方法
    """
    train_seq = scaled_log_return_series_history[-seq_len:]
    garch_model_instance = None

    # 根据模型类型选择合适的配置
    if 'GARCH-M' in model_type:
        try:
            if model_type == 'GARCH-M(1,1)_AR':
                # 第一步：使用更稳定的GARCH模型配置
                base_model = arch_model(train_seq, 
                                     mean='Zero', 
                                     vol='GARCH', 
                                     p=1, 
                                     q=1,
                                     dist='skewt')  # 使用偏t分布以更好地处理非对称性
                
                # 使用更稳健的拟合设置
                base_res = base_model.fit(disp='off', 
                                        show_warning=False,
                                        update_freq=5, 
                                        cov_type='robust',
                                        options={'maxiter': 2000},  # 增加最大迭代次数
                                        tol=1e-8)  # 提高收敛精度
                
                # 获取条件波动率并进行预处理
                conditional_vol = base_res.conditional_volatility
                if np.any(np.isnan(conditional_vol)) or np.any(np.isinf(conditional_vol)):
                    raise ValueError("条件波动率计算出现无效值")
                
                # 使用更稳健的波动率变换方法
                log_vol = np.log1p(np.clip(conditional_vol, 1e-10, None))  # 使用log1p避免log(0)
                winsorized_vol = np.clip(log_vol, 
                                       np.percentile(log_vol, 1),
                                       np.percentile(log_vol, 99))
                scaled_vol = (winsorized_vol - np.mean(winsorized_vol)) / (np.std(winsorized_vol) + 1e-8)
                transformed_vol = np.tanh(scaled_vol * 0.2)  # 使用tanh限制范围在[-1,1]
                
                # 创建外生变量矩阵
                x_for_arx = pd.DataFrame({'vol': transformed_vol}, 
                                       index=train_seq.index[-len(transformed_vol):])
                
                # 拟合ARX模型
                arx_model = arch_model(train_seq, 
                                     x=x_for_arx, 
                                     mean='ARX', 
                                     lags=p_ar,
                                     vol='Constant', 
                                     dist='skewt')
                
                # 使用更稳健的拟合设置
                arx_res = arx_model.fit(disp='off', 
                                      show_warning=False,
                                      update_freq=5, 
                                      cov_type='robust',
                                      options={'maxiter': 2000},
                                      tol=1e-8)
                
                # 使用蒙特卡洛模拟进行多步预测
                n_sims = 1000
                vol_forecasts = base_res.forecast(horizon=pred_len, 
                                                method='simulation',
                                                simulations=n_sims)
                predicted_std_dev = np.sqrt(vol_forecasts.variance.values[-1, :pred_len])
                
                # 对预测的波动率进行同样的变换
                log_pred_vol = np.log1p(np.clip(predicted_std_dev, 1e-10, None))
                winsorized_pred_vol = np.clip(log_pred_vol,
                                            np.percentile(log_vol, 1),
                                            np.percentile(log_vol, 99))
                scaled_pred_vol = (winsorized_pred_vol - np.mean(log_vol)) / (np.std(log_vol) + 1e-8)
                transformed_pred_vol = np.tanh(scaled_pred_vol * 0.2)
                
                # 创建预测用的外生变量矩阵
                x_forecast_df = pd.DataFrame({'vol': transformed_pred_vol}, 
                                           index=np.arange(pred_len))
                
                # 使用稳健的方法预测均值
                mean_forecasts = []
                last_obs = np.array(train_seq[-p_ar:] if p_ar > 0 else [])
                
                # 获取模型参数并应用正则化
                const_param = arx_res.params.get('const', 0)
                ar_params = [arx_res.params.get(f'ar[{i+1}]', 0) for i in range(p_ar)]
                vol_param = arx_res.params.get('vol', 0)
                
                # 使用自适应风险溢价系数
                prior_vol_param = 0.02  # 减小先验风险溢价系数
                posterior_vol_param = (vol_param + prior_vol_param) / 2
                vol_param = np.clip(posterior_vol_param, -0.1, 0.1)  # 进一步限制范围
                
                for h in range(pred_len):
                    mean_forecast = const_param
                    
                    # AR项，使用指数衰减权重
                    if p_ar > 0:
                        for i in range(min(p_ar, len(last_obs))):
                            decay_factor = np.exp(-0.1 * i)  # 减小衰减速度
                            ar_effect = ar_params[i] * last_obs[-(i+1)] * decay_factor
                            ar_effect = np.clip(ar_effect, -1.0, 1.0)  # 收紧限制
                            mean_forecast += ar_effect
                    
                    # 风险溢价效应，使用softplus函数限制效应
                    vol_effect = vol_param * np.log1p(np.exp(transformed_pred_vol[h]))
                    mean_forecast += vol_effect
                    
                    # 使用softplus函数限制最终预测值
                    mean_forecast = np.log1p(np.exp(mean_forecast)) - np.log(2)
                    mean_forecasts.append(mean_forecast)
                    
                    # 更新用于下一步预测的观测值
                    if len(last_obs) > 0:
                        noise = np.random.normal(0, 0.001)  # 减小随机噪声
                        last_obs = np.append(last_obs[1:], mean_forecast + noise)
                
                # 最终预测值限制在合理范围内
                predictions = np.clip(mean_forecasts, -3.0, 3.0)
                return predictions
                
            else:  # GARCH-M(1,1)_Const
                # 使用更稳定的GARCH-M配置
                base_garch = arch_model(train_seq, 
                                      mean='Constant', 
                                      vol='GARCH', 
                                      p=1, 
                                      q=1,
                                      dist='skewt')
                
                base_res = base_garch.fit(disp='off', 
                                         show_warning=False,
                                         update_freq=5, 
                                         cov_type='robust',
                                         options={'maxiter': 2000},
                                         tol=1e-8)
                
                const_param = base_res.params['mu']
                
                # 使用蒙特卡洛模拟预测波动率
                n_sims = 1000
                vol_forecasts = base_res.forecast(horizon=pred_len, 
                                                method='simulation',
                                                simulations=n_sims)
                predicted_std_dev = np.sqrt(vol_forecasts.variance.values[-1, :pred_len])
                
                # 对预测的波动率进行稳健变换
                log_pred_vol = np.log1p(np.clip(predicted_std_dev, 1e-10, None))
                winsorized_log_pred_vol = np.clip(log_pred_vol,
                                                np.percentile(log_pred_vol, 1),
                                                np.percentile(log_pred_vol, 99))
                scaled_pred_vol = (winsorized_log_pred_vol - np.mean(log_pred_vol)) / (np.std(log_pred_vol) + 1e-8)
                transformed_pred_vol = np.tanh(scaled_pred_vol * 0.2)
                
                # 使用自适应风险溢价系数
                prior_premium = 0.02
                sample_premium = base_res.params.get('mu', 0.02)
                posterior_premium = (prior_premium + sample_premium) / 2
                risk_premium_coef = np.clip(posterior_premium, -0.1, 0.1)
                
                # GARCH-M预测，使用softplus函数
                garch_m_forecasts = []
                for vol in transformed_pred_vol:
                    premium_effect = risk_premium_coef * np.log1p(np.exp(vol))
                    forecast = const_param + premium_effect
                    forecast = np.clip(forecast, -3.0, 3.0)  # 限制预测范围
                    garch_m_forecasts.append(forecast)
                
                return np.array(garch_m_forecasts)
                
        except Exception as e_garch_m:
            print(f"警告: {model_type} 模型拟合或预测失败 ({str(e_garch_m)})。尝试使用纯ARMA模型。")
            try:
                arima_model = ARIMA(train_seq, order=(p_ar, 0, q_ma))
                arima_results = arima_model.fit()
                forecast_val = arima_results.forecast(steps=pred_len)
                return np.array(forecast_val.values if isinstance(forecast_val, pd.Series) else forecast_val).flatten()[:pred_len]
            except Exception as e_arima:
                print(f"警告: {model_type}的纯ARMA({p_ar},0,{q_ma})回退失败: {e_arima}。返回零预测。")
                return np.zeros(pred_len)
    else: 
        # 根据模型类型和均值设置选择合适的配置
        if model_type == 'ARCH(1)_AR':
            garch_model_instance = arch_model(train_seq, mean='AR', lags=p_ar, vol='ARCH', p=1, q=0)
        elif model_type == 'ARCH(1)_Const':
            garch_model_instance = arch_model(train_seq, mean='Constant', vol='ARCH', p=1, q=0)
        elif model_type == 'GARCH(1,1)_AR':
            garch_model_instance = arch_model(train_seq, mean='AR', lags=p_ar, vol='GARCH', p=1, o=0, q=1)
        elif model_type == 'GARCH(1,1)_Const':
            garch_model_instance = arch_model(train_seq, mean='Constant', vol='GARCH', p=1, o=0, q=1)
        elif model_type == 'TARCH(1,1)_AR' or model_type == 'ZARCH(1,1)_AR':
            # TARCH/ZARCH 模型 (GJR-GARCH)
            garch_model_instance = arch_model(train_seq, mean='AR', lags=p_ar, vol='GARCH', p=1, o=1, q=1, power=1.0)
        elif model_type == 'TARCH(1,1)_Const' or model_type == 'ZARCH(1,1)_Const':
            garch_model_instance = arch_model(train_seq, mean='Constant', vol='GARCH', p=1, o=1, q=1, power=1.0)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        try:
            fit_update_freq = 1
            results = garch_model_instance.fit(disp='off', show_warning=False, update_freq=fit_update_freq)
            
            # 使用模拟方法进行多步预测
            forecasts = results.forecast(horizon=pred_len, method='simulation', simulations=1000)
            predictions = forecasts.mean.values[-1, :pred_len]
            
            # 限制预测值范围
            predictions = np.clip(predictions, -3.0, 3.0)
            return predictions
            
        except Exception as e_garch_fit:
            print(f"警告: {model_type} 模型拟合或预测失败 ({str(e_garch_fit)})。尝试使用纯ARMA模型。")
            try:
                arima_model = ARIMA(train_seq, order=(p_ar, 0, q_ma))
                arima_results = arima_model.fit()
                forecast_val = arima_results.forecast(steps=pred_len)
                return np.array(forecast_val.values if isinstance(forecast_val, pd.Series) else forecast_val).flatten()[:pred_len]
            except Exception as e_arima_fallback:
                print(f"警告: {model_type}的纯ARMA({p_ar},0,{q_ma})回退失败: {e_arima_fallback}。返回零预测。")
                return np.zeros(pred_len)

def train_and_forecast_pure_arma(scaled_log_return_series_history, seq_len, pred_len, p_ar=1, q_ma=0):
    """
    训练纯ARMA模型并进行多步预测
    """
    train_seq = scaled_log_return_series_history[-seq_len:]
    
    arima_model = ARIMA(train_seq, order=(p_ar, 0, q_ma))
    try:
        arima_results = arima_model.fit()
        forecast_val = arima_results.forecast(steps=pred_len)
        if isinstance(forecast_val, pd.Series): # Handles Series output
            forecast_val = forecast_val.values
        # Ensure it's a 1D array matching pred_len
        return np.array(forecast_val).flatten()[:pred_len]
    except Exception as e_arima:
        print(f"警告: ARMA({p_ar},0,{q_ma}) 模型拟合失败: {e_arima}。返回零预测。")
        return np.zeros(pred_len)

def rolling_forecast_multistep(data_df, original_price_col_name, log_return_col_name, 
                     model_type, seq_len, pred_len, step_size=1, p=1, q=0, test_df=None):
    """
    对测试集执行多步滚动预测
    
    参数:
    data_df: 包含原始价格和对数收益率的DataFrame
    original_price_col_name: 原始价格列名
    log_return_col_name: 对数收益率列名
    model_type: 模型类型
    seq_len: 用于训练的序列长度
    pred_len: 预测步数
    step_size: 滚动窗口步长
    p: AR项阶数
    q: MA项阶数
    test_df: 测试集DataFrame
    
    返回:
    真实价格, 预测价格, 额外信息
    """
    if test_df is None:
        # 使用完整数据集
        log_returns = data_df[log_return_col_name].values
        original_prices = data_df[original_price_col_name].values
    else:
        # 使用训练集+验证集来拟合标准化器，但仅使用测试集进行评估
        log_returns = data_df[log_return_col_name].values
        original_prices = test_df[original_price_col_name].values

    # 标准化对数收益率
    log_return_scaler = StandardScaler()
    scaled_log_returns = log_return_scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
    
    # 获取测试集起始位置
    if test_df is None:
        # 如果没有提供测试集，使用完整数据的最后部分
        num_total_log_points = len(scaled_log_returns)
        test_size = int(0.15 * num_total_log_points)  # 使用15%作为测试集
        test_start_idx = num_total_log_points - test_size
    else:
        # 如果提供了测试集，则使用训练数据 + 测试集前seq_len个点进行预测
        test_start_idx = len(log_returns) - len(test_df)
    
    if test_start_idx < 0:
        print(f"警告: 模型 {model_type} 计算的 test_start_idx ({test_start_idx}) < 0。没有足够的测试数据。")
        return np.array([]), np.array([]), None

    # 用于收集预测结果的列表
    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []
    
    # 极端值过滤阈值
    extreme_value_threshold = 5.0

    # 执行滚动预测
    for i in tqdm(range(test_start_idx, len(scaled_log_returns) - pred_len + 1, step_size),
                 desc=f"{model_type} 预测进度 ({pred_len}天)", file=sys.stderr):
        
        # 使用到当前点的所有历史数据进行训练
        current_history_for_model_input = scaled_log_returns[:i]
        # 模型内部会选择最后seq_len个点
        train_seq_actually_used_by_model = current_history_for_model_input[-seq_len:]

        # 获取用于重构价格的最后一个实际价格
        if test_df is None:
            last_actual_price_for_reconstruction = original_prices[i-1]
        else:
            # 如果使用单独的测试集，确保索引正确映射
            test_idx = i - test_start_idx - 1
            if test_idx >= 0:
                last_actual_price_for_reconstruction = original_prices[test_idx]
            else:
                # 可能需要从训练集获取最后一个价格
                last_actual_price_for_reconstruction = data_df[original_price_col_name].values[i-1]

        # 执行多步预测
        predicted_std_log_returns_raw = train_and_forecast_arima_garch_multistep(
            current_history_for_model_input, model_type, seq_len, pred_len, p_ar=p, q_ma=q
        )

        # 检查预测结果是否有效
        use_arma_fallback_in_rolling = False

        if not isinstance(predicted_std_log_returns_raw, np.ndarray) or \
           predicted_std_log_returns_raw.shape != (pred_len,):
            print(f"警告: {model_type} (在滚动窗口索引 {i}, 训练序列实际使用长度 {len(train_seq_actually_used_by_model)}) 返回了格式/形状错误的预测: {predicted_std_log_returns_raw}.")
            use_arma_fallback_in_rolling = True
        elif np.any(np.isnan(predicted_std_log_returns_raw)) or \
             np.any(np.isinf(predicted_std_log_returns_raw)):
            print(f"警告: {model_type} (在滚动窗口索引 {i}, 训练序列实际使用长度 {len(train_seq_actually_used_by_model)}) 返回了 NaN/Inf 预测: {predicted_std_log_returns_raw}.")
            use_arma_fallback_in_rolling = True
        elif np.any(np.abs(predicted_std_log_returns_raw) > extreme_value_threshold):
            print(f"警告: {model_type} (在滚动窗口索引 {i}, 训练序列实际使用长度 {len(train_seq_actually_used_by_model)}) 返回了极端值预测 (abs > {extreme_value_threshold}): {predicted_std_log_returns_raw}.")
            if len(train_seq_actually_used_by_model) > 0:
                print(f"       触发极端值的训练序列摘要: len={len(train_seq_actually_used_by_model)}, mean={np.mean(train_seq_actually_used_by_model):.4f}, std={np.std(train_seq_actually_used_by_model):.4f}, min={np.min(train_seq_actually_used_by_model):.4f}, max={np.max(train_seq_actually_used_by_model):.4f}")
            else:
                print(f"       触发极端值的训练序列为空或长度不足 (传递给模型的序列长度为 {len(current_history_for_model_input)}, 模型内部截取最后 {seq_len})。")
            use_arma_fallback_in_rolling = True

        # 如果GARCH预测有问题，回退到ARMA模型
        if use_arma_fallback_in_rolling:
            print(f"       此步骤 ({model_type} @索引 {i}) 回退到 Pure ARMA({p},{q})。")
            predicted_std_log_returns = train_and_forecast_pure_arma(
                current_history_for_model_input, seq_len, pred_len, p_ar=p, q_ma=q # Pure ARMA also takes full history and slices
            )
            if not (isinstance(predicted_std_log_returns, np.ndarray) and \
                    predicted_std_log_returns.shape == (pred_len,) and \
                    not np.any(np.isnan(predicted_std_log_returns)) and \
                    not np.any(np.isinf(predicted_std_log_returns)) and \
                    not np.any(np.abs(predicted_std_log_returns) > extreme_value_threshold)):
                print(f"警告: Pure ARMA({p},{q}) 回退的预测仍然无效/极端 ({model_type} @索引 {i}): {predicted_std_log_returns}. 使用零预测代替。")
                predicted_std_log_returns = np.zeros(pred_len)
        else:
            predicted_std_log_returns = predicted_std_log_returns_raw
        
        # 反标准化预测的对数收益率
        predicted_log_returns_for_window = log_return_scaler.inverse_transform(
            np.array(predicted_std_log_returns).reshape(-1, 1)
        ).flatten()

        # 基于对数收益率预测重构价格
        reconstructed_price_forecasts_this_window = []
        current_reconstructed_price = last_actual_price_for_reconstruction
        for log_ret_pred_step in predicted_log_returns_for_window:
            log_ret_pred_step = np.clip(log_ret_pred_step, -5, 5)  # 限制极端值
            current_reconstructed_price = current_reconstructed_price * np.exp(log_ret_pred_step)
            reconstructed_price_forecasts_this_window.append(current_reconstructed_price)
        
        # 获取实际价格
        if test_df is None:
            actual_price_levels_this_window = original_prices[i : i + pred_len]
        else:
            # 如果使用单独的测试集，确保索引正确映射
            test_idx = i - test_start_idx
            if test_idx >= 0:
                actual_end = min(test_idx + pred_len, len(original_prices))
                actual_price_levels_this_window = original_prices[test_idx : actual_end]
                if len(actual_price_levels_this_window) < pred_len:
                    # 如果实际价格不足pred_len个，则跳过此样本
                    continue
            else:
                # 超出范围，跳过
                continue
        
        # 收集结果
        all_predicted_price_levels_collected.append(reconstructed_price_forecasts_this_window)
        all_true_price_levels_collected.append(actual_price_levels_this_window)

    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([]), None 

    return np.concatenate(all_true_price_levels_collected), np.concatenate(all_predicted_price_levels_collected), None

def rolling_forecast_pure_arma(data_df, original_price_col_name, log_return_col_name, 
                     seq_len, pred_len, step_size=1, p=1, q=0, test_df=None):
    """
    对测试集执行纯ARMA模型的多步滚动预测
    
    参数与rolling_forecast_multistep相同，但使用纯ARMA模型
    """
    if test_df is None:
        # 使用完整数据集
        log_returns = data_df[log_return_col_name].values
        original_prices = data_df[original_price_col_name].values
    else:
        # 使用训练集+验证集来拟合标准化器，但仅使用测试集进行评估
        log_returns = data_df[log_return_col_name].values
        original_prices = test_df[original_price_col_name].values

    # 标准化对数收益率
    log_return_scaler = StandardScaler()
    scaled_log_returns = log_return_scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
    
    # 获取测试集起始位置
    if test_df is None:
        # 如果没有提供测试集，使用完整数据的最后部分
        num_total_log_points = len(scaled_log_returns)
        test_size = int(0.15 * num_total_log_points)  # 使用15%作为测试集
        test_start_idx = num_total_log_points - test_size
    else:
        # 如果提供了测试集，则使用训练数据 + 测试集前seq_len个点进行预测
        test_start_idx = len(log_returns) - len(test_df)
    
    if test_start_idx < 0:
        print(f"警告: 纯ARMA模型计算的 test_start_idx ({test_start_idx}) < 0。没有足够的测试数据。")
        return np.array([]), np.array([]), None

    # 用于收集预测结果的列表
    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []
    
    # 极端值过滤阈值
    extreme_value_threshold = 5.0

    # 执行滚动预测
    for i in tqdm(range(test_start_idx, len(scaled_log_returns) - pred_len + 1, step_size),
                 desc=f"Pure ARMA({p},{q}) 预测进度 ({pred_len}天)", file=sys.stderr):
        
        # 使用到当前点的所有历史数据进行训练
        current_history_for_model_input = scaled_log_returns[:i]
        
        # 获取用于重构价格的最后一个实际价格
        if test_df is None:
            last_actual_price_for_reconstruction = original_prices[i-1]
        else:
            # 如果使用单独的测试集，确保索引正确映射
            test_idx = i - test_start_idx - 1
            if test_idx >= 0:
                last_actual_price_for_reconstruction = original_prices[test_idx]
            else:
                # 可能需要从训练集获取最后一个价格
                last_actual_price_for_reconstruction = data_df[original_price_col_name].values[i-1]

        # 执行纯ARMA多步预测
        predicted_std_log_returns = train_and_forecast_pure_arma(
            current_history_for_model_input, seq_len, pred_len, p_ar=p, q_ma=q
        )
        
        # 检查预测结果是否有效
        if not (isinstance(predicted_std_log_returns, np.ndarray) and \
                predicted_std_log_returns.shape == (pred_len,) and \
                not np.any(np.isnan(predicted_std_log_returns)) and \
                not np.any(np.isinf(predicted_std_log_returns)) and \
                not np.any(np.abs(predicted_std_log_returns) > extreme_value_threshold)):
            print(f"警告: Pure ARMA({p},{q}) (在滚动窗口索引 {i}) 返回了无效/极端预测: {predicted_std_log_returns}. 使用零预测代替。")
            predicted_std_log_returns = np.zeros(pred_len)
        
        # 反标准化预测的对数收益率
        predicted_log_returns_for_window = log_return_scaler.inverse_transform(
            np.array(predicted_std_log_returns).reshape(-1, 1)
        ).flatten()

        # 基于对数收益率预测重构价格
        reconstructed_price_forecasts_this_window = []
        current_reconstructed_price = last_actual_price_for_reconstruction
        for log_ret_pred_step in predicted_log_returns_for_window:
            log_ret_pred_step = np.clip(log_ret_pred_step, -5, 5)  # 限制极端值
            current_reconstructed_price = current_reconstructed_price * np.exp(log_ret_pred_step)
            reconstructed_price_forecasts_this_window.append(current_reconstructed_price)
        
        # 获取实际价格
        if test_df is None:
            actual_price_levels_this_window = original_prices[i : i + pred_len]
        else:
            # 如果使用单独的测试集，确保索引正确映射
            test_idx = i - test_start_idx
            if test_idx >= 0:
                actual_end = min(test_idx + pred_len, len(original_prices))
                actual_price_levels_this_window = original_prices[test_idx : actual_end]
                if len(actual_price_levels_this_window) < pred_len:
                    # 如果实际价格不足pred_len个，则跳过此样本
                    continue
            else:
                # 超出范围，跳过
                continue
        
        # 收集结果
        all_predicted_price_levels_collected.append(reconstructed_price_forecasts_this_window)
        all_true_price_levels_collected.append(actual_price_levels_this_window)

    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([]), None 

    return np.concatenate(all_true_price_levels_collected), np.concatenate(all_predicted_price_levels_collected), None

def run_naive_baseline_forecast(data_df, original_price_col_name, pred_len, step_size, test_df=None):
    """
    运行朴素基线模型（上一时间步的值）进行多步预测
    """
    if test_df is None:
        # 使用完整数据集
        original_prices = data_df[original_price_col_name].values
        num_total_points = len(original_prices)
        test_size = int(0.15 * num_total_points)  # 使用15%作为测试集
        test_start_idx = num_total_points - test_size
    else:
        # 使用提供的测试集
        original_prices = test_df[original_price_col_name].values
        test_start_idx = 0

    if test_start_idx < 0:
        print(f"警告: 数据点总数不足以支持测试集划分。")
        return np.array([]), np.array([])

    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []

    for i in tqdm(range(test_start_idx, len(original_prices) - pred_len + 1, step_size), 
                 desc=f"Naive Baseline 预测进度 ({pred_len}天)", file=sys.stderr):
        if i < 1: continue 
        
        # 获取实际价格序列
        actual_price_levels_this_window = original_prices[i : i + pred_len] 
        
        # 使用前一天的值作为所有未来时间步的预测
        value_from_previous_day = original_prices[i-1] 
        predicted_price_levels_this_window = [value_from_previous_day] * pred_len
        
        all_true_price_levels_collected.append(actual_price_levels_this_window)
        all_predicted_price_levels_collected.append(predicted_price_levels_this_window)
        
        if len(actual_price_levels_this_window) < pred_len: 
            break 

    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([])

    return np.concatenate(all_true_price_levels_collected), np.concatenate(all_predicted_price_levels_collected)

def plot_results(results_dict, pred_len, results_dir_path, config_label):
    """
    绘制预测结果并保存图像，包含置信区间和多步预测效果的可视化
    
    参数:
    results_dict: 包含各模型预测结果的字典
    pred_len: 预测步数
    results_dir_path: 结果保存目录
    config_label: 配置标签
    """
    # 按模型类型排序
    sorted_model_keys = []
    
    # 优先排序朴素基线模型
    if "Naive Baseline (PrevDay)" in results_dict: 
        sorted_model_keys.append("Naive Baseline (PrevDay)")
    
    # 确保Pure ARMA排在其他GARCH族模型前面
    other_keys = [k for k in results_dict.keys() if k != "Naive Baseline (PrevDay)"]
    pure_arma_keys = sorted([k for k in other_keys if "Pure ARMA" in k])
    sorted_model_keys.extend(pure_arma_keys)
    
    # 添加其余模型
    garch_keys = sorted([k for k in other_keys if k not in pure_arma_keys])
    sorted_model_keys.extend(garch_keys)

    num_models_to_plot = len(sorted_model_keys)
    if num_models_to_plot == 0: 
        return
    
    # 为每个模型创建两个子图：一个总体视图，一个局部多步预测视图
    plt.figure(figsize=(20, 8 * num_models_to_plot)) 
    
    for plot_idx, model_type_key in enumerate(sorted_model_keys):
        result_data = results_dict.get(model_type_key)
        if not (result_data and 'true_values' in result_data and 'predictions' in result_data and \
                len(result_data['true_values']) > 0 and len(result_data['predictions']) > 0):
            print(f"警告: 模型 {model_type_key} 数据不完整，跳过绘图。")
            continue
            
        true_vals = np.asarray(result_data['true_values']).flatten()
        pred_vals = np.asarray(result_data['predictions']).flatten()
        
        # 计算预测的置信区间（使用移动标准差作为不确定性估计）
        window_size = min(pred_len * 3, len(true_vals))
        rolling_std = pd.Series(np.abs(true_vals - pred_vals)).rolling(window=window_size, center=True).std()
        confidence_factor = 1.96  # 95% 置信区间
        lower_bound = pred_vals - confidence_factor * rolling_std
        upper_bound = pred_vals + confidence_factor * rolling_std
        
        # 总体视图
        plt.subplot(num_models_to_plot, 2, 2*plot_idx + 1)
        
        # 计算要显示的点数
        max_pts = min(1000, len(true_vals))
        display_step = max(1, max_pts // 100)  # 确保不会显示太多点
        
        # 绘制置信区间
        plt.fill_between(range(len(pred_vals[:max_pts:display_step])),
                        lower_bound[:max_pts:display_step],
                        upper_bound[:max_pts:display_step],
                        color='gray', alpha=0.2, label='95% 置信区间')
        
        # 绘制实际值和预测值
        plt.plot(true_vals[:max_pts:display_step], 
                label='实际价格', color='blue', marker='o', markersize=3)
        plt.plot(pred_vals[:max_pts:display_step], 
                label='预测价格', color='red', linestyle='--', marker='x', markersize=3)
        
        plt.title(f'{model_type_key} ({config_label})\n整体预测效果', fontsize=12)
        plt.xlabel('时间步长 (测试集样本)', fontsize=10)
        plt.ylabel('汇率', fontsize=10)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 局部多步预测视图
        plt.subplot(num_models_to_plot, 2, 2*plot_idx + 2)
        
        # 选择一个代表性的预测窗口进行展示
        window_start = len(true_vals) // 2  # 从中间位置开始
        window_size = pred_len * 3  # 显示3个预测周期
        
        # 绘制局部预测窗口的置信区间
        plt.fill_between(range(window_start, window_start + window_size),
                        lower_bound[window_start:window_start + window_size],
                        upper_bound[window_start:window_start + window_size],
                        color='gray', alpha=0.2, label='95% 置信区间')
        
        # 绘制局部实际值和预测值
        plt.plot(range(window_start, window_start + window_size),
                true_vals[window_start:window_start + window_size],
                label='实际价格', color='blue', marker='o')
        plt.plot(range(window_start, window_start + window_size),
                pred_vals[window_start:window_start + window_size],
                label='预测价格', color='red', linestyle='--', marker='x')
        
        # 标记预测步长
        for i in range(window_start, window_start + window_size - pred_len + 1, pred_len):
            plt.axvspan(i, i + pred_len, color='yellow', alpha=0.1)
            plt.axvline(x=i, color='green', linestyle=':', alpha=0.5)
            if i == window_start:
                plt.annotate(f'预测步长: {pred_len}天',
                           xy=(i, plt.ylim()[0]),
                           xytext=(i, plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0])*0.1),
                           arrowprops=dict(facecolor='black', shrink=0.05),
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        plt.title(f'{model_type_key} ({config_label})\n局部多步预测效果展示', fontsize=12)
        plt.xlabel('时间步长', fontsize=10)
        plt.ylabel('汇率', fontsize=10)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加说明文字
        plt.figtext(0.02, 0.98 - (plot_idx)/num_models_to_plot, 
                   f"说明:\n"
                   f"1. 蓝线(实线+圆点): 实际价格\n"
                   f"2. 红线(虚线+叉号): 预测价格\n"
                   f"3. 灰色区域: 95% 置信区间\n"
                   f"4. 黄色区域: {pred_len}天预测窗口\n"
                   f"5. 绿色虚线: 预测起始点",
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_path, f'plot_price_level_{config_label}_pred_len_{pred_len}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(all_results_summary, results_dir_path, config_label):
    """
    生成汇总表格，展示不同模型在各预测步长上的性能
    
    all_results_summary: 格式为 {pred_len: {model_name: {metrics: {}, time: ...}}} 的字典
    """
    # 初始化汇总数据
    summary_data = {'Model': []}
    metric_names = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'Time(s)'] 
    
    pred_lengths_present = sorted(all_results_summary.keys())
    if not pred_lengths_present: 
        return pd.DataFrame()

    # 初始化列
    for pred_len_val in pred_lengths_present: 
        for metric_n in metric_names: 
            summary_data[f'{metric_n}_{pred_len_val}'] = [] 
    
    # 确定模型顺序
    model_types_present_set = set()
    for pred_len_val in pred_lengths_present:
        if isinstance(all_results_summary[pred_len_val], dict):
            model_types_present_set.update(all_results_summary[pred_len_val].keys())
    
    # 按照特定顺序排列模型（Naive -> ARMA -> ARCH/GARCH族）
    ordered_model_types = []
    if "Naive Baseline (PrevDay)" in model_types_present_set: 
        ordered_model_types.append("Naive Baseline (PrevDay)")
    
    # 添加Pure ARMA模型
    pure_arma_keys = sorted([k for k in model_types_present_set if "Pure ARMA" in k])
    ordered_model_types.extend(pure_arma_keys)

    # 添加其他GARCH族模型
    other_garch_models = sorted([k for k in model_types_present_set if k not in ordered_model_types])
    ordered_model_types.extend(other_garch_models)

    if not ordered_model_types: 
        return pd.DataFrame()

    # 填充数据
    for model_t in ordered_model_types:
        summary_data['Model'].append(model_t)
        for pred_len_val in pred_lengths_present:
            model_result = None
            
            # 查找当前模型在当前预测步长的结果
            if pred_len_val in all_results_summary and model_t in all_results_summary[pred_len_val]:
                model_result = all_results_summary[pred_len_val][model_t]
            
            # 填充每个指标列
            for metric_n in metric_names:
                col_name = f'{metric_n}_{pred_len_val}'
                
                if model_result and isinstance(model_result, dict):
                    if metric_n == 'Time(s)' and 'time' in model_result:
                        # 时间列，保留3位小数
                        formatted_val = f"{model_result['time']:.3f}"
                    elif 'metrics' in model_result and metric_n in model_result['metrics']:
                        # 性能指标列
                        val = model_result['metrics'][metric_n]
                        if metric_n == 'MAPE':
                            # MAPE已经是百分比，保留4位小数
                            formatted_val = f"{val:.4f}%"
                        elif metric_n == 'R2':
                            # R2保留6位小数
                            formatted_val = f"{val:.6f}"
                        else:
                            # 其他指标保留8位小数
                            formatted_val = f"{val:.8f}"
                    else:
                        formatted_val = 'N/A'
                else:
                    formatted_val = 'N/A'
                
                summary_data[col_name].append(formatted_val)
    
    # 创建DataFrame并保存
    df_summary = pd.DataFrame(summary_data)
    
    # 设置pandas显示选项，显示所有列和更多小数位
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: '%.8f' if isinstance(x, float) else str(x))
    
    # 保存为CSV时不使用科学计数法
    csv_path = os.path.join(results_dir_path, f'summary_table_{config_label}.csv')
    df_summary.to_csv(csv_path, index=False, float_format='%.8f')
    
    print(f"\n结果汇总 ({config_label} - 按0.7/0.15/0.15比例划分训练/验证/测试):")
    print(df_summary.to_string())
    
    return df_summary

def main(args):
    """
    主函数：加载数据，训练模型，进行预测，评估性能，保存结果
    """
    # 对数收益率列名
    log_return_col = 'log_return_usd_jpy' 

    try:
        # 加载数据并计算对数收益率
        data_df = load_and_prepare_data(args.root_path, args.data_path, args.target, log_return_col)
    except ValueError as e:
        print(f"数据加载或预处理失败: {e}")
        return

    # 执行数据集划分
    train_df, val_df, test_df = split_data(data_df, train_ratio=0.7, val_ratio=0.15)
    
    # 确认训练集大小足够
    if len(train_df) < args.seq_len:
        print(f"错误: 训练集大小 ({len(train_df)}) 小于模型序列长度 ({args.seq_len})。请减少 seq_len 或增加数据。")
        return

    # 对对数收益率进行平稳性检验
    if not data_df[log_return_col].empty:
        perform_stationarity_test(data_df[log_return_col], series_name=f"{args.target} 的对数收益率 (USDJPY)")
    else:
        print("对数收益率序列为空。")
        return

    # 定义模型类型
    # 区分AR均值和Constant均值的模型
    ar_mean_models = [
        'ARCH(1)_AR', 'GARCH(1,1)_AR',
        'TARCH(1,1)_AR',
        'GARCH-M(1,1)_AR'
    ]
    
    constant_mean_models = [
        'ARCH(1)_Const', 'GARCH(1,1)_Const',
        'TARCH(1,1)_Const',
        'GARCH-M(1,1)_Const'
    ]
    
    # 预测步长列表：1天、7天、30天、60天、90天、180天
    pred_lens = [1]
    
    # ARMA参数组合
    arma_params = [(1, 0), (1, 1), (2, 0)]
    
    # 序列长度组合
    seq_lens_to_test = [args.seq_len]
    
    best_overall_metrics = {'MSE': float('inf'), 'model': None, 'p': None, 'q': None, 'seq_len': None, 'pred_len': None, 'metrics': None}
    all_configs_comparison = {}
    
    for current_seq_len in seq_lens_to_test:
        print(f"\n{'#'*80}")
        print(f"测试滚动窗口大小 (seq_len): {current_seq_len}")
        print(f"{'#'*80}")
    
        if len(train_df) < current_seq_len:
            print(f"警告: 初始训练数据点 ({len(train_df)}) 少于当前序列长度 ({current_seq_len})。跳过此序列长度。")
            continue

        # 首先运行朴素基线模型
        baseline_results_dir = f'arima_garch_results_logret_USDJPY_baseline_seq{current_seq_len}'
        os.makedirs(baseline_results_dir, exist_ok=True)
        
        baseline_run_results = {}
        baseline_metrics_by_pred_len = {pred_len: {} for pred_len in pred_lens}
        
        print(f"\n{'-'*50}")
        print(f"模型: Naive Baseline (PrevDay)")
        print(f"{'-'*50}")
        
        for pred_len in pred_lens:
            print(f"预测步长: {pred_len}")
            start_time_naive = time.time()
            naive_actuals, naive_preds = run_naive_baseline_forecast(
                data_df, args.target, pred_len, args.step_size, test_df=test_df
            )
            elapsed_time_naive = time.time() - start_time_naive

            if len(naive_actuals) > 0:
                eval_metrics_naive = evaluate_model(naive_actuals, naive_preds)
                baseline_run_results[pred_len] = {
                    'metrics': eval_metrics_naive, 'true_values': naive_actuals,
                    'predictions': naive_preds, 'time': elapsed_time_naive
                }
                baseline_metrics_by_pred_len[pred_len]["Naive Baseline (PrevDay)"] = baseline_run_results[pred_len]
                print(f"执行时间: {elapsed_time_naive:.2f}秒")
                for name, val_metric in eval_metrics_naive.items():
                    print(f"{name}: {val_metric:.4f}{'%' if name == 'MAPE' else ''}")
        
        # 运行Constant均值模型(只需运行一次，不依赖ARMA参数)
        print(f"\n{'='*80}")
        print(f"运行Constant均值模型组 (seq_len={current_seq_len})")
        print(f"{'='*80}")
        
        constant_results_dir = f'arima_garch_results_logret_USDJPY_constant_seq{current_seq_len}'
        os.makedirs(constant_results_dir, exist_ok=True)
        
        constant_run_results = {}
        constant_metrics_by_pred_len = {pred_len: {} for pred_len in pred_lens}
        
        # 添加朴素基线模型结果到constant_run_results
        if baseline_run_results:
            constant_run_results["Naive Baseline (PrevDay)"] = baseline_run_results
            with open(os.path.join(constant_results_dir, f'naive_baseline_USDJPY_constant_seq{current_seq_len}.pkl'), 'wb') as f:
                pickle.dump(baseline_run_results, f)
            
            # 将朴素基线结果也添加到constant_metrics
            for pred_len in pred_lens:
                if pred_len in baseline_run_results:
                    constant_metrics_by_pred_len[pred_len]["Naive Baseline (PrevDay)"] = baseline_run_results[pred_len]

        # Constant均值模型
        for model_type in constant_mean_models:
            model_run_results = {}
            for pred_len in pred_lens:
                print(f"\n{'-'*50}")
                print(f"模型: {model_type} - 预测步长: {pred_len}")
                print(f"{'-'*50}")
                start_time = time.time()
                actuals, preds, _ = rolling_forecast_multistep(
                    data_df, args.target, log_return_col, model_type,
                    current_seq_len, pred_len, args.step_size, p=1, q=0, test_df=test_df
                )
                elapsed_time = time.time() - start_time
                
                if len(actuals) > 0:
                    eval_metrics = evaluate_model(actuals, preds)
                    model_run_results[pred_len] = {
                        'metrics': eval_metrics, 'true_values': actuals,
                        'predictions': preds, 'time': elapsed_time
                    }
                    constant_metrics_by_pred_len[pred_len][model_type] = model_run_results[pred_len]
                    print(f"执行时间: {elapsed_time:.2f}秒")
                    for name, val_metric in eval_metrics.items():
                        print(f"{name}: {val_metric:.4f}{'%' if name == 'MAPE' else ''}")
                    
                    if eval_metrics['MSE'] < best_overall_metrics['MSE']:
                        best_overall_metrics.update({
                            'MSE': eval_metrics['MSE'],
                            'model': model_type,
                            'p': None,
                            'q': None,
                            'seq_len': current_seq_len,
                            'pred_len': pred_len,
                            'metrics': eval_metrics
                        })
                
            # 保存每个模型的所有预测步长结果
            if model_run_results:
                constant_run_results[model_type] = model_run_results
                with open(os.path.join(constant_results_dir, f'{model_type}_USDJPY_constant_seq{current_seq_len}.pkl'), 'wb') as f:
                    pickle.dump(model_run_results, f)
        
        # 保存Constant均值模型结果摘要并生成图表
        if constant_run_results:
            for pred_len in pred_lens:
                plot_data = {}
                for model_name, model_results in constant_run_results.items():
                    if pred_len in model_results:
                        plot_data[model_name] = model_results[pred_len]
                
                if plot_data:
                    plot_results(plot_data, pred_len, constant_results_dir, f"constant_seq{current_seq_len}")
            
            # 生成汇总表
            generate_summary_table(constant_metrics_by_pred_len, constant_results_dir, f"constant_seq{current_seq_len}")

        # 然后运行AR均值模型(需要循环ARMA参数)
        for p_val, q_val in arma_params:
            print(f"\n{'='*80}")
            print(f"测试AR均值模型组 ARMA(p={p_val}, q={q_val}) (seq_len={current_seq_len})")
            print(f"{'='*80}")
            
            arima_params_label_str = f"p{p_val}q{q_val}_seq{current_seq_len}"
            current_results_dir = f'arima_garch_results_logret_USDJPY_{arima_params_label_str}'
            os.makedirs(current_results_dir, exist_ok=True)
            
            run_results_by_model = {}
            metrics_by_pred_len = {pred_len: {} for pred_len in pred_lens}

            # 添加朴素基线模型结果到当前ARMA参数的结果中
            if baseline_run_results:
                run_results_by_model["Naive Baseline (PrevDay)"] = baseline_run_results
                with open(os.path.join(current_results_dir, f'naive_baseline_USDJPY_{arima_params_label_str}.pkl'), 'wb') as f:
                    pickle.dump(baseline_run_results, f)
                
                # 将朴素基线结果也添加到metrics
                for pred_len in pred_lens:
                    if pred_len in baseline_run_results:
                        metrics_by_pred_len[pred_len]["Naive Baseline (PrevDay)"] = baseline_run_results[pred_len]

            # Pure ARMA
            arma_model_name_label = f'Pure ARMA({p_val},{q_val})'
            arma_run_results = {}
            
            for pred_len in pred_lens:
                print(f"\n{'-'*50}")
                print(f"模型: {arma_model_name_label} - 预测步长: {pred_len}")
                print(f"{'-'*50}")
                start_time_arma = time.time()
                arma_actuals, arma_preds, _ = rolling_forecast_pure_arma(
                    data_df, args.target, log_return_col,
                    current_seq_len, pred_len, args.step_size, p_val, q_val, test_df=test_df
                )
                elapsed_time_arma = time.time() - start_time_arma
                
                if len(arma_actuals) > 0:
                    eval_metrics_arma = evaluate_model(arma_actuals, arma_preds)
                    arma_run_results[pred_len] = {
                        'metrics': eval_metrics_arma, 'true_values': arma_actuals,
                        'predictions': arma_preds, 'time': elapsed_time_arma
                    }
                    metrics_by_pred_len[pred_len][arma_model_name_label] = arma_run_results[pred_len]
                    print(f"执行时间: {elapsed_time_arma:.2f}秒")
                    for name, val_metric in eval_metrics_arma.items():
                        print(f"{name}: {val_metric:.4f}{'%' if name == 'MAPE' else ''}")
                    
                    if eval_metrics_arma['MSE'] < best_overall_metrics['MSE']:
                        best_overall_metrics.update({
                            'MSE': eval_metrics_arma['MSE'],
                            'model': arma_model_name_label,
                            'p': p_val,
                            'q': q_val,
                            'seq_len': current_seq_len,
                            'pred_len': pred_len,
                            'metrics': eval_metrics_arma
                        })
            
            # 保存Pure ARMA结果
            if arma_run_results:
                run_results_by_model[arma_model_name_label] = arma_run_results
                with open(os.path.join(current_results_dir, f'{arma_model_name_label}_USDJPY_{arima_params_label_str}.pkl'), 'wb') as f:
                    pickle.dump(arma_run_results, f)

            # AR均值的GARCH族模型
            for model_type in ar_mean_models:
                model_label = f"{model_type} (AR p={p_val}, MA q={q_val})"
                model_run_results = {}
                
                for pred_len in pred_lens:
                    print(f"\n{'-'*50}")
                    print(f"模型: {model_type} (AR p={p_val}, MA q={q_val}) - 预测步长: {pred_len}")
                    print(f"{'-'*50}")
                    start_time = time.time()
                    actuals, preds, _ = rolling_forecast_multistep(
                        data_df, args.target, log_return_col, model_type,
                        current_seq_len, pred_len, args.step_size, p_val, q_val, test_df=test_df
                    )
                    elapsed_time = time.time() - start_time
                    
                    if len(actuals) > 0:
                        eval_metrics = evaluate_model(actuals, preds)
                        model_run_results[pred_len] = {
                            'metrics': eval_metrics, 'true_values': actuals,
                            'predictions': preds, 'time': elapsed_time
                        }
                        metrics_by_pred_len[pred_len][model_label] = model_run_results[pred_len]
                        print(f"执行时间: {elapsed_time:.2f}秒")
                        for name, val_metric in eval_metrics.items():
                            print(f"{name}: {val_metric:.4f}{'%' if name == 'MAPE' else ''}")
                        
                        if eval_metrics['MSE'] < best_overall_metrics['MSE']:
                            best_overall_metrics.update({
                                'MSE': eval_metrics['MSE'],
                                'model': f'{model_type} (AR({p_val}))',
                                'p': p_val,
                                'q': q_val,
                                'seq_len': current_seq_len,
                                'pred_len': pred_len,
                                'metrics': eval_metrics
                            })
                
                # 保存每个模型的结果
                if model_run_results:
                    run_results_by_model[model_label] = model_run_results
                    with open(os.path.join(current_results_dir, f'{model_type}_USDJPY_{arima_params_label_str}.pkl'), 'wb') as f:
                        pickle.dump(model_run_results, f)

            # 保存当前ARMA参数组合的所有模型结果
            if run_results_by_model:
                # 为每个预测步长生成图表
                for pred_len in pred_lens:
                    plot_data = {}
                    for model_name, model_results in run_results_by_model.items():
                        if pred_len in model_results:
                            plot_data[model_name] = model_results[pred_len]
                    
                    if plot_data:
                        plot_results(plot_data, pred_len, current_results_dir, arima_params_label_str)
                
                # 生成汇总表
                generate_summary_table(metrics_by_pred_len, current_results_dir, arima_params_label_str)

            # 更新配置比较结果
            config_comparison = {}
            for pred_len in pred_lens:
                if metrics_by_pred_len[pred_len]:
                    best_model_entries = [(model_name, model_data['metrics']['MSE']) 
                                         for model_name, model_data in metrics_by_pred_len[pred_len].items()
                                         if isinstance(model_data, dict) and 'metrics' in model_data and 'MSE' in model_data['metrics']]
                    
                    if best_model_entries:
                        best_model = min(best_model_entries, key=lambda x: x[1])
                        config_comparison[pred_len] = {
                            'best_model_name': best_model[0],
                            'best_mse': best_model[1],
                            'all_metrics_in_config': metrics_by_pred_len[pred_len]
                        }
            
            if config_comparison:
                all_configs_comparison[arima_params_label_str] = config_comparison

        # 将Constant均值模型的结果也添加到配置比较中
        constant_config_label = f"constant_seq{current_seq_len}"
        config_comparison = {}
        for pred_len in pred_lens:
            if constant_metrics_by_pred_len[pred_len]:
                best_model_entries = [(model_name, model_data['metrics']['MSE']) 
                                     for model_name, model_data in constant_metrics_by_pred_len[pred_len].items()
                                     if isinstance(model_data, dict) and 'metrics' in model_data and 'MSE' in model_data['metrics']]
                
                if best_model_entries:
                    best_model = min(best_model_entries, key=lambda x: x[1])
                    config_comparison[pred_len] = {
                        'best_model_name': best_model[0],
                        'best_mse': best_model[1],
                        'all_metrics_in_config': constant_metrics_by_pred_len[pred_len]
                    }
        
        if config_comparison:
            all_configs_comparison[constant_config_label] = config_comparison

    # 打印总结
    print("\n" + "="*80)
    print("所有配置的总结:")
    print("="*80)
    for config_label, summary_res in all_configs_comparison.items():
        print(f"\n配置: {config_label}")
        for pred_len in pred_lens:
            if pred_len in summary_res:
                best_model_info = summary_res[pred_len]
                print(f"  预测步长 {pred_len}: 最佳模型 = {best_model_info['best_model_name']} (MSE: {best_model_info['best_mse']:.4f})")
    
    print("\n" + "="*80)
    print("全局最佳模型配置:")
    if best_overall_metrics['metrics']:
        print(f"模型类型: {best_overall_metrics['model']}")
        if best_overall_metrics['p'] is not None:
            print(f"ARMA参数: p={best_overall_metrics['p']}, q={best_overall_metrics['q']}")
        print(f"滚动窗口大小: {best_overall_metrics['seq_len']}")
        print(f"预测步长: {best_overall_metrics['pred_len']}")
        print("\n评估指标:")
        for metric_name, value in best_overall_metrics['metrics'].items():
            print(f"  {metric_name}: {value:.4f}{'%' if metric_name == 'MAPE' else ''}")
    else:
        print("未能找到最佳模型。")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARIMA+GARCH汇率多步预测 (USDJPY, 对数收益率, 多步预测)')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='数据根目录')
    parser.add_argument('--data_path', type=str, default='sorted_output_file.csv', help='数据文件路径')
    parser.add_argument('--target', type=str, default='rate', help='原始汇率目标变量列名')
    parser.add_argument('--seq_len', type=int, default=144, help='ARIMA+GARCH历史对数收益率长度')
    parser.add_argument('--step_size', type=int, default=1
    , help='滚动窗口步长')
    
    args = parser.parse_args()
    main(args) 