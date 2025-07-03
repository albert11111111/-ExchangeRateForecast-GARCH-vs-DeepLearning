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
from scipy import stats
import seaborn as sns
import statsmodels.api as sm

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
    df.reset_index(drop=True, inplace=True)
    
    if df.empty:
        raise ValueError("计算对数收益率并移除NaN后，数据为空。")
        
    return df

def perform_stationarity_test(series, series_name="序列"):
    """对给定序列执行ADF平稳性检验并打印结果。"""
    print(f"\n对 {series_name} 进行ADF平稳性检验:")
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
    
    # 计算预测方向准确率
    direction_accuracy = np.mean((np.diff(y_true_prices) * np.diff(y_pred_prices)) > 0) * 100
    
    # 计算Theil's U统计量
    changes_actual = np.diff(y_true_prices) / y_true_prices[:-1]
    changes_pred = np.diff(y_pred_prices) / y_pred_prices[:-1]
    
    numerator = np.sqrt(np.mean((changes_actual - changes_pred) ** 2))
    denominator = np.sqrt(np.mean(changes_actual ** 2)) + np.sqrt(np.mean(changes_pred ** 2))
    theils_u = numerator / denominator if denominator != 0 else np.nan
    
    # 计算预测偏差比率
    bias_ratio = np.mean(y_pred_prices - y_true_prices) / np.mean(y_true_prices) * 100
    
    return {
        'MSE': mse, 
        'RMSE': rmse, 
        'MAE': mae, 
        'MAPE': mape, 
        'R2': r2,
        'Direction_Accuracy': direction_accuracy,
        'Theils_U': theils_u,
        'Bias_Ratio': bias_ratio
    }

def split_data_fixed_test(data_df, test_size=150):
    """
    将数据集划分为训练集和测试集，使用固定数量的最后n个点作为测试集
    """
    if len(data_df) <= test_size:
        raise ValueError(f"数据集大小 ({len(data_df)}) 小于或等于要求的测试集大小 ({test_size})。")
    
    train_df = data_df.iloc[:-test_size].copy()
    test_df = data_df.iloc[-test_size:].copy()
    
    print(f"数据集总大小: {len(data_df)}")
    print(f"训练集大小: {len(train_df)} ({len(train_df)/len(data_df):.2%})")
    print(f"测试集大小: {len(test_df)} ({len(test_df)/len(data_df):.2%})")
    
    return train_df, test_df

def print_model_diagnostics(model_results, model_type):
    """
    打印模型诊断信息
    
    参数:
    model_results: 模型拟合结果
    model_type: 模型类型
    """
    print(f"\n{'-'*50}")
    print(f"模型诊断信息 - {model_type}")
    print(f"{'-'*50}")
    
    try:
        # 打印模型参数
        print("\n模型参数:")
        for param, value in model_results.params.items():
            print(f"{param:15s}: {value:10.4f} (std: {model_results.std_err.get(param, np.nan):10.4f})")
        
        # 打印模型拟合信息
        print("\n拟合信息:")
        print(f"对数似然值: {model_results.loglikelihood:10.4f}")
        print(f"AIC: {model_results.aic:10.4f}")
        print(f"BIC: {model_results.bic:10.4f}")
        
        # 打印收敛信息
        print("\n收敛信息:")
        if hasattr(model_results, 'convergence_flag'):
            print(f"收敛标志: {model_results.convergence_flag}")
        if hasattr(model_results, 'num_iter'):
            print(f"迭代次数: {model_results.num_iter}")
        
        # 打印残差统计信息
        resid = model_results.resid
        print("\n残差统计:")
        print(f"均值: {np.mean(resid):10.4f}")
        print(f"标准差: {np.std(resid):10.4f}")
        print(f"偏度: {stats.skew(resid):10.4f}")
        print(f"峰度: {stats.kurtosis(resid):10.4f}")
        
        # 进行Ljung-Box检验
        lb_test = sm.stats.diagnostic.acorr_ljungbox(resid, lags=[10, 20, 30], return_df=True)
        print("\nLjung-Box检验结果:")
        print(lb_test.to_string())
        
    except Exception as e:
        print(f"\n警告: 诊断信息生成失败 - {str(e)}")
    
    print(f"\n{'-'*50}\n")

def train_and_forecast_arima_garch_multistep(scaled_log_return_series_history, model_type, seq_len, pred_len, p_ar=1, q_ma=0):
    """
    训练ARIMA-GARCH模型并进行多步预测
    
    使用arch库的forecast功能进行多步预测，针对不同模型使用适当的预测方法
    """
    # 确保train_seq是pandas Series，这对于GARCH-M模型很重要
    train_seq = pd.Series(scaled_log_return_series_history[-seq_len:])
    
    # 数据有效性检查
    if np.any(np.isnan(train_seq)) or np.any(np.isinf(train_seq)):
        print(f"警告: 输入序列包含无效值 (NaN/Inf)，尝试清理...")
        train_seq = pd.Series(np.nan_to_num(train_seq.values, nan=0, posinf=None, neginf=None))
    
    # 异常值处理
    z_scores = np.abs((train_seq - train_seq.mean()) / train_seq.std())
    train_seq[z_scores > 3] = np.sign(train_seq[z_scores > 3]) * (3 * train_seq.std())
    
    garch_model_instance = None

    # 根据模型类型选择合适的配置
    if 'GARCH-M' in model_type:
        try:
            if model_type == 'GARCH-M(1,1)_AR':
                # 第一步：使用更稳定的GARCH模型配置
                try:
                    base_model = arch_model(train_seq, 
                                         mean='Zero', 
                                         vol='GARCH', 
                                         p=1, 
                                         q=1,
                                         dist='skewt')
                except Exception as e_model:
                    print(f"警告: GARCH-M模型创建失败 ({str(e_model)})，尝试使用更简单的配置...")
                    base_model = arch_model(train_seq,
                                         mean='Zero',
                                         vol='GARCH',
                                         p=1,
                                         q=1,
                                         dist='normal')
                
                # 使用更稳健的拟合设置，添加重试机制
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        base_res = base_model.fit(disp='off', 
                                                show_warning=False,
                                                update_freq=5, 
                                                cov_type='robust',
                                                options={'maxiter': 2000},
                                                tol=1e-8)
                        break
                    except Exception as e_fit:
                        if attempt < max_attempts - 1:
                            print(f"警告: 第{attempt+1}次拟合尝试失败 ({str(e_fit)})，重试...")
                            continue
                        else:
                            raise Exception(f"所有拟合尝试都失败了: {str(e_fit)}")
                
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
                
                # 创建外生变量矩阵，确保index正确对齐
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
                
                # 打印模型诊断信息
                print_model_diagnostics(base_res, f"{model_type} - 基础GARCH")
                print_model_diagnostics(arx_res, f"{model_type} - ARX")
                
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
                                           index=pd.RangeIndex(pred_len))
                
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
                
                # 打印模型诊断信息
                print_model_diagnostics(base_res, model_type)
                
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

def calculate_prediction_intervals(predicted_values, std_dev, confidence_level=0.95):
    """
    计算预测区间
    
    参数:
    predicted_values: 预测值序列
    std_dev: 标准差序列
    confidence_level: 置信水平
    
    返回:
    lower_bounds, upper_bounds: 下界和上界
    """
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # 使用指数加权移动标准差
    if isinstance(std_dev, (float, int)):
        std_dev = np.ones_like(predicted_values) * std_dev
    elif len(std_dev) == 1:
        std_dev = np.ones_like(predicted_values) * std_dev[0]
    
    # 计算区间
    margin = z_score * std_dev
    lower_bounds = predicted_values - margin
    upper_bounds = predicted_values + margin
    
    # 确保区间合理（非负）
    if np.any(predicted_values > 0):
        lower_bounds = np.maximum(lower_bounds, 0)
    
    return lower_bounds, upper_bounds

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
        # 固定使用最后150个点作为测试集
        test_size = 150
        test_start_idx = len(log_returns) - test_size
    else:
        # 使用训练集+验证集来拟合标准化器，但仅使用测试集进行评估
        log_returns = data_df[log_return_col_name].values
        original_prices = test_df[original_price_col_name].values
        test_start_idx = len(log_returns) - len(test_df)

    # 标准化对数收益率
    log_return_scaler = StandardScaler()
    scaled_log_returns = log_return_scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
    
    if test_start_idx < 0:
        print(f"警告: 模型 {model_type} 计算的 test_start_idx ({test_start_idx}) < 0。没有足够的测试数据。")
        return np.array([]), np.array([]), None

    # 用于收集预测结果的列表
    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []
    all_prediction_intervals = []
    
    # 计算历史波动率
    historical_std = np.std(scaled_log_returns)
    
    # 执行滚动预测
    for i in tqdm(range(test_start_idx, len(scaled_log_returns) - pred_len + 1, step_size),
                 desc=f"{model_type} 预测进度 ({pred_len}天)", file=sys.stderr):
        
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

        # 执行多步预测
        predicted_std_log_returns_raw = train_and_forecast_arima_garch_multistep(
            current_history_for_model_input, model_type, seq_len, pred_len, p_ar=p, q_ma=q
        )

        # 检查预测结果是否有效
        use_arma_fallback_in_rolling = False

        if not isinstance(predicted_std_log_returns_raw, np.ndarray) or \
           predicted_std_log_returns_raw.shape != (pred_len,):
            print(f"警告: {model_type} (在滚动窗口索引 {i}) 返回了格式/形状错误的预测: {predicted_std_log_returns_raw}.")
            use_arma_fallback_in_rolling = True
        elif np.any(np.isnan(predicted_std_log_returns_raw)) or \
             np.any(np.isinf(predicted_std_log_returns_raw)):
            print(f"警告: {model_type} (在滚动窗口索引 {i}) 返回了 NaN/Inf 预测: {predicted_std_log_returns_raw}.")
            use_arma_fallback_in_rolling = True
        elif np.any(np.abs(predicted_std_log_returns_raw) > 5.0):
            print(f"警告: {model_type} (在滚动窗口索引 {i}) 返回了极端值预测 (abs > 5.0): {predicted_std_log_returns_raw}.")
            use_arma_fallback_in_rolling = True

        # 如果GARCH预测有问题，回退到ARMA模型
        if use_arma_fallback_in_rolling:
            print(f"       此步骤 ({model_type} @索引 {i}) 回退到 Pure ARMA({p},{q})。")
            predicted_std_log_returns = train_and_forecast_pure_arma(
                current_history_for_model_input, seq_len, pred_len, p_ar=p, q_ma=q
            )
            if not (isinstance(predicted_std_log_returns, np.ndarray) and \
                    predicted_std_log_returns.shape == (pred_len,) and \
                    not np.any(np.isnan(predicted_std_log_returns)) and \
                    not np.any(np.isinf(predicted_std_log_returns)) and \
                    not np.any(np.abs(predicted_std_log_returns) > 5.0)):
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
        
        # 使用改进的预测区间计算
        window_returns = []
        window_prices = []
        prediction_intervals = []
        
        for step, log_ret_pred_step in enumerate(predicted_log_returns_for_window):
            # 限制极端值
            log_ret_pred_step = np.clip(log_ret_pred_step, -5, 5)
            window_returns.append(log_ret_pred_step)
            
            # 计算价格
            current_reconstructed_price = current_reconstructed_price * np.exp(log_ret_pred_step)
            reconstructed_price_forecasts_this_window.append(current_reconstructed_price)
            window_prices.append(current_reconstructed_price)
            
            # 计算该步的预测区间
            if step == 0:
                # 第一步使用历史波动率
                step_std = historical_std
            else:
                # 后续步骤使用预测窗口内的波动率
                step_std = np.std(window_returns)
            
            lower, upper = calculate_prediction_intervals(
                np.array([current_reconstructed_price]),
                step_std * np.sqrt(step + 1)  # 考虑预测步长的累积不确定性
            )
            prediction_intervals.append((lower[0], upper[0]))
        
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
        all_prediction_intervals.append(prediction_intervals)

    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([]), None 

    return (np.concatenate(all_true_price_levels_collected), 
            np.concatenate(all_predicted_price_levels_collected), 
            np.array(all_prediction_intervals).reshape(-1, 2))

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
        test_size = 150  # 固定使用最后150个点作为测试集
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
    all_prediction_intervals = []
    
    # 计算历史波动率
    historical_std = np.std(scaled_log_returns)
    
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
                not np.any(np.abs(predicted_std_log_returns) > 5.0)):
            print(f"警告: Pure ARMA({p},{q}) (在滚动窗口索引 {i}) 返回了无效/极端预测: {predicted_std_log_returns}. 使用零预测代替。")
            predicted_std_log_returns = np.zeros(pred_len)
        
        # 反标准化预测的对数收益率
        predicted_log_returns_for_window = log_return_scaler.inverse_transform(
            np.array(predicted_std_log_returns).reshape(-1, 1)
        ).flatten()

        # 基于对数收益率预测重构价格
        reconstructed_price_forecasts_this_window = []
        current_reconstructed_price = last_actual_price_for_reconstruction
        
        # 使用改进的预测区间计算
        window_returns = []
        window_prices = []
        prediction_intervals = []
        
        for step, log_ret_pred_step in enumerate(predicted_log_returns_for_window):
            # 限制极端值
            log_ret_pred_step = np.clip(log_ret_pred_step, -5, 5)
            window_returns.append(log_ret_pred_step)
            
            # 计算价格
            current_reconstructed_price = current_reconstructed_price * np.exp(log_ret_pred_step)
            reconstructed_price_forecasts_this_window.append(current_reconstructed_price)
            window_prices.append(current_reconstructed_price)
            
            # 计算该步的预测区间
            if step == 0:
                # 第一步使用历史波动率
                step_std = historical_std
            else:
                # 后续步骤使用预测窗口内的波动率
                step_std = np.std(window_returns)
            
            lower, upper = calculate_prediction_intervals(
                np.array([current_reconstructed_price]),
                step_std * np.sqrt(step + 1)  # 考虑预测步长的累积不确定性
            )
            prediction_intervals.append((lower[0], upper[0]))
        
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
        all_prediction_intervals.append(prediction_intervals)

    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([]), None 

    return (np.concatenate(all_true_price_levels_collected), 
            np.concatenate(all_predicted_price_levels_collected), 
            np.array(all_prediction_intervals).reshape(-1, 2))

def run_naive_baseline_forecast(data_df, original_price_col_name, pred_len, step_size, test_df=None):
    """
    运行朴素基线模型（上一时间步的值）进行多步预测
    """
    if test_df is None:
        # 使用完整数据集
        original_prices = data_df[original_price_col_name].values
        num_total_points = len(original_prices)
        test_size = 150  # 固定使用最后150个点作为测试集
        test_start_idx = num_total_points - test_size
        test_prices = original_prices[-test_size:]
    else:
        # 使用提供的测试集
        test_prices = test_df[original_price_col_name].values
        test_start_idx = 0

    if len(test_prices) < pred_len:
        print(f"警告: 测试集数据点数量 ({len(test_prices)}) 少于预测步长 ({pred_len})。")
        return np.array([]), np.array([])

    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []

    # 获取训练集最后一个值用于第一个预测窗口
    if test_df is not None:
        last_train_price = data_df[original_price_col_name].values[-1]
    else:
        last_train_price = original_prices[test_start_idx-1]

    for i in tqdm(range(0, len(test_prices) - pred_len + 1, step_size), 
                 desc=f"Naive Baseline 预测进度 ({pred_len}天)", file=sys.stderr):
        # 获取实际价格序列
        actual_price_levels_this_window = test_prices[i : i + pred_len]
        
        # 使用前一天的值作为所有未来时间步的预测
        if i == 0:
            value_from_previous_day = last_train_price
        else:
            value_from_previous_day = test_prices[i-1]
            
        predicted_price_levels_this_window = [value_from_previous_day] * pred_len
        
        all_true_price_levels_collected.append(actual_price_levels_this_window)
        all_predicted_price_levels_collected.append(predicted_price_levels_this_window)

    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([])

    return np.concatenate(all_true_price_levels_collected), np.concatenate(all_predicted_price_levels_collected)

def main(args):
    log_return_col = 'log_return_usd_jpy' 
    fixed_test_set_size = 150
    try:
        data_df = load_and_prepare_data(args.root_path, args.data_path, args.target, log_return_col)
    except ValueError as e:
        print(f"数据加载或预处理失败: {e}")
        return
    if len(data_df) <= fixed_test_set_size:
        print(f"错误: 数据点总数 ({len(data_df)}) 不足以分割出 {fixed_test_set_size} 点的测试集。")
        return
    
    # 划分训练集和测试集
    train_df, test_df = split_data_fixed_test(data_df, fixed_test_set_size)
    
    num_total_points = len(data_df)
    num_train_points_for_first_window = num_total_points - fixed_test_set_size
    if num_train_points_for_first_window < args.seq_len:
        print(f"错误: 初始训练数据点 ({num_train_points_for_first_window}) 少于模型序列长度 ({args.seq_len})。请减少 seq_len 或增加数据。")
        return
    if not data_df[log_return_col].empty:
        perform_stationarity_test(data_df[log_return_col], series_name=f"{args.target} 的对数收益率 (USDJPY)")
    else:
        print("对数收益率序列为空。")
        return
    constant_mean_models = [
        'ARCH(1)_Const', 'GARCH(1,1)_Const', 'EGARCH(1,1)_Const', 'GARCH-M(1,1)_Const'
    ]
    pred_lens = [1, 7, 14, 30, 60, 90, 150]
    seq_lens_to_test = [48, 96, 144, 192]
    best_overall_metrics = {'MSE': float('inf'), 'model': None, 'p': None, 'q': None, 'seq_len': None, 'metrics': None}
    all_configs_comparison = {}
    for current_seq_len in seq_lens_to_test:
        args.seq_len = current_seq_len
        print(f"\n{'#'*80}")
        print(f"测试滚动窗口大小 (seq_len): {current_seq_len}")
        print(f"{'#'*80}")
        if num_train_points_for_first_window < current_seq_len:
            print(f"警告: 初始训练数据点 ({num_train_points_for_first_window}) 少于当前序列长度 ({current_seq_len})。跳过此序列长度。")
            continue
        print(f"\n{'='*80}")
        print(f"运行Constant均值模型组 (seq_len={current_seq_len})")
        print(f"{'='*80}")
        constant_results_dir = f'arima_garch_results_logret_USDJPY_constant_seq{current_seq_len}_multistep_last{fixed_test_set_size}test'
        os.makedirs(constant_results_dir, exist_ok=True)
        constant_run_results = {}
        constant_metrics = {}
        print(f"\n{'-'*50}\n模型: Naive Baseline (PrevDay)\n{'-'*50}")
        start_time_naive = time.time()
        naive_actuals, naive_preds = run_naive_baseline_forecast(
            data_df, args.target, pred_lens[0], args.step_size, test_df=test_df
        )
        elapsed_time_naive = time.time() - start_time_naive
        if len(naive_actuals) > 0:
            eval_metrics_naive = evaluate_model(naive_actuals, naive_preds)
            constant_run_results["Naive Baseline (PrevDay)"] = {
                'metrics': eval_metrics_naive, 'true_values': naive_actuals,
                'predictions': naive_preds, 'time': elapsed_time_naive
            }
            constant_metrics["Naive Baseline (PrevDay)"] = eval_metrics_naive
            print(f"执行时间: {elapsed_time_naive:.2f}秒")
            for name, val_metric in eval_metrics_naive.items():
                print(f"{name}: {val_metric:.4f}{'%' if name == 'MAPE' else ''}")
        for model_type in constant_mean_models:
            for pred_len in pred_lens:
                print(f"\n{'-'*50}\n模型: {model_type} (预测步长={pred_len})\n{'-'*50}")
                start_time = time.time()
                actuals, preds, _ = rolling_forecast_multistep(
                    data_df, args.target, log_return_col, model_type,
                    current_seq_len, pred_len, args.step_size, p=1, q=0, test_df=test_df
                )
                elapsed_time = time.time() - start_time
                if len(actuals) > 0:
                    eval_metrics = evaluate_model(actuals, preds)
                    key = f"{model_type}_step{pred_len}"
                    constant_run_results[key] = {
                        'metrics': eval_metrics, 'true_values': actuals,
                        'predictions': preds, 'time': elapsed_time
                    }
                    constant_metrics[key] = eval_metrics
                    print(f"执行时间: {elapsed_time:.2f}秒")
                    for name, val_metric in eval_metrics.items():
                        print(f"{name}: {val_metric:.4f}{'%' if name == 'MAPE' else ''}")
                    if eval_metrics['MSE'] < best_overall_metrics['MSE']:
                        best_overall_metrics.update({
                            'MSE': eval_metrics['MSE'],
                            'model': key,
                            'p': None,
                            'q': None,
                            'seq_len': current_seq_len,
                            'metrics': eval_metrics
                        })
        if constant_run_results:
            plot_results(constant_run_results, pred_lens[0], constant_results_dir, f"constant_seq{current_seq_len}_multistep")
            with open(os.path.join(constant_results_dir, f'results_USDJPY_constant_seq{current_seq_len}_multistep.pkl'), 'wb') as f:
                pickle.dump(constant_run_results, f)
            summary_input_for_table = {pred_lens[0]: constant_run_results}
            generate_summary_table(summary_input_for_table, constant_results_dir, f"constant_seq{current_seq_len}_multistep")
        constant_config_label = f"constant_seq{current_seq_len}_multistep"
        if constant_metrics:
            all_configs_comparison[constant_config_label] = {
                'best_model_name': min(constant_metrics, key=lambda k: constant_metrics[k]['MSE']),
                'best_mse': min(m['MSE'] for m in constant_metrics.values()),
                'all_metrics_in_config': constant_metrics
            }
    print("\n" + "="*80)
    print("所有配置的总结:")
    print("="*80)
    for config_label, summary_res in all_configs_comparison.items():
        print(f"\n配置: {config_label}")
        print(f"  此配置中最佳模型: {summary_res['best_model_name']} (MSE: {summary_res['best_mse']:.4f})")
    print("\n" + "="*80)
    print("全局最佳模型配置:")
    if best_overall_metrics['metrics']:
        print(f"模型类型组合: {best_overall_metrics['model']}")
        print(f"滚动窗口大小: {best_overall_metrics['seq_len']}")
        print("\n评估指标:")
        for metric_name, value in best_overall_metrics['metrics'].items():
            print(f"  {metric_name}: {value:.4f}{'%' if metric_name == 'MAPE' else ''}")
    else:
        print("未能找到最佳模型。")
    print("="*80)

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
    
    # 为每个模型创建多个子图
    num_plots_per_model = 4  # 总体视图、局部多步预测视图、误差分布图、误差自相关图
    plt.figure(figsize=(20, 8 * num_models_to_plot * (num_plots_per_model // 2)))
    
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
        
        base_plot_idx = plot_idx * num_plots_per_model
        
        # 1. 总体视图
        plt.subplot(num_models_to_plot * 2, 2, base_plot_idx + 1)
        
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
        
        # 2. 局部多步预测视图
        plt.subplot(num_models_to_plot * 2, 2, base_plot_idx + 2)
        
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
        
        # 3. 误差分布图
        plt.subplot(num_models_to_plot * 2, 2, base_plot_idx + 3)
        
        # 计算预测误差
        errors = pred_vals - true_vals
        
        # 绘制误差直方图和核密度估计
        sns.histplot(errors, kde=True, stat='density', bins=50)
        
        # 添加正态分布拟合曲线
        mu, std = stats.norm.fit(errors)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, label=f'正态分布拟合\n(μ={mu:.4f}, σ={std:.4f})')
        
        # 添加统计信息
        skewness = stats.skew(errors)
        kurtosis = stats.kurtosis(errors)
        plt.text(0.05, 0.95, f'偏度: {skewness:.4f}\n峰度: {kurtosis:.4f}',
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title(f'{model_type_key} ({config_label})\n预测误差分布', fontsize=12)
        plt.xlabel('预测误差', fontsize=10)
        plt.ylabel('密度', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 4. 误差自相关图
        plt.subplot(num_models_to_plot * 2, 2, base_plot_idx + 4)
        
        # 计算误差的自相关系数
        nlags = min(40, len(errors) - 1)
        acf = sm.tsa.acf(errors, nlags=nlags)
        
        # 绘制自相关图
        plt.bar(range(len(acf)), acf, alpha=0.5)
        plt.axhline(y=0, linestyle='-', color='black', linewidth=0.5)
        
        # 添加置信区间
        conf_int = 1.96 / np.sqrt(len(errors))
        plt.axhline(y=conf_int, linestyle='--', color='red', alpha=0.5)
        plt.axhline(y=-conf_int, linestyle='--', color='red', alpha=0.5)
        
        # 添加Ljung-Box检验结果
        lb_test = sm.stats.diagnostic.acorr_ljungbox(errors, lags=[10], return_df=True)
        lb_stat = lb_test.iloc[0]['lb_stat']
        lb_pvalue = lb_test.iloc[0]['lb_pvalue']
        plt.text(0.05, 0.95, f'Ljung-Box检验 (lag=10):\n统计量: {lb_stat:.4f}\np值: {lb_pvalue:.4f}',
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title(f'{model_type_key} ({config_label})\n预测误差自相关', fontsize=12)
        plt.xlabel('滞后阶数', fontsize=10)
        plt.ylabel('自相关系数', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加说明文字
        plt.figtext(0.02, 0.98 - (plot_idx * 2)/num_models_to_plot, 
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
    metric_names = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'Direction_Accuracy', 'Theils_U', 'Bias_Ratio', 'Time(s)']
    
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
                        if metric_n in ['MAPE', 'Direction_Accuracy', 'Bias_Ratio']:
                            # 这些指标是百分比，保留4位小数
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
    
    print(f"\n结果汇总 ({config_label} - 固定最后150个点作为测试集):")
    print(df_summary.to_string())
    
    return df_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARIMA+GARCH多步预测 (USDJPY, 对数收益率, 多步, 最后150点测试)')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='数据根目录')
    parser.add_argument('--data_path', type=str, default='sorted_output_file.csv', help='数据文件路径')
    parser.add_argument('--target', type=str, default='rate', help='原始汇率目标变量列名')
    parser.add_argument('--seq_len', type=int, default=144, help='ARIMA+GARCH历史对数收益率长度 (会被循环覆盖)')
    parser.add_argument('--step_size', type=int, default=1, help='滚动窗 口步长')
    args = parser.parse_args()
    main(args) 