#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ARIMA+GARCH时间序列预测系统 - GBP/CNY汇率预测版本
================================================================

功能说明:
--------
这是一个专门用于英镑兑人民币汇率预测的ARIMA+GARCH模型系统，支持多种预测步长。

主要特性:
--------
1. 多步预测支持: 1天、24天、30天、60天、90天、180天预测
2. 多种模型类型:
   - Naive Baseline: 朴素基线模型
   - Pure ARMA: 纯时间序列模型  
   - GARCH族模型: ARCH、GARCH、EGARCH、GARCH-M、GJR-GARCH、TGARCH
3. 两种均值模型:
   - Constant均值: 使用常数均值
   - AR均值: 使用自回归均值
4. 滚动窗口预测: 使用数据集的最后15%作为测试集
5. 全面诊断: Ljung-Box检验、正态性检验、残差分析

输出结果:
--------
1. 预测结果图表 (plot_price_level_*.png)
2. 模型参数文件 (model_params_*.json)
3. 完整结果数据 (results_*.pkl)
4. 评估指标汇总表 (summary_table_*.csv)

使用方法:
--------
python run_arima_garch_gbp_cny_dynamic_test.py --root_path ./dataset/ --data_path 英镑兑人民币_short_series.csv --target Close

作者: 时间序列预测系统
版本: GBP/CNY动态测试集版 v2.0
日期: 2024
"""

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
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import json

# 设置中文字体
try:
    font = FontProperties(fname=r"C:\Windows\Fonts\SimHei.ttf")
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告：无法加载中文字体，图表中的中文可能显示为方框")

# 忽略警告
warnings.filterwarnings('ignore')

def load_and_prepare_data(root_path, data_path, target_col_name, log_return_col_name='log_return'):
    """
    数据加载和预处理函数
    
    功能说明:
    --------
    1. 加载汇率数据文件
    2. 处理日期列并排序
    3. 计算对数收益率 (log return)
    4. 数据清洗和验证
    
    参数:
    ----
    root_path : str
        数据文件根目录路径
    data_path : str  
        数据文件名
    target_col_name : str
        目标变量列名 (如'Close'表示汇率)
    log_return_col_name : str, 默认='log_return'
        对数收益率列名
        
    返回:
    ----
    pd.DataFrame
        包含日期、原始汇率和对数收益率的清洗后DataFrame
        
    异常:
    ----
    ValueError
        当目标列不存在、数据类型错误或处理后数据为空时抛出
        
    数学原理:
    --------
    对数收益率计算: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    其中 P_t 为 t 时刻的价格，r_t 为 t 时刻的对数收益率
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
    adf_result = adfuller(series.dropna()) 
    print(f'\n对 {series_name} 进行ADF平稳性检验:')
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



def perform_model_diagnostics(residuals, model_name="Unknown"):
    try:
        stats_info = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals, fisher=True) + 3),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'n_samples': len(residuals)
        }
        
        normality_tests = {
            'jarque_bera': {
                'statistic': float(stats.jarque_bera(residuals)[0]),
                'p_value': float(stats.jarque_bera(residuals)[1])
            },
            'shapiro': {
                'statistic': float(stats.shapiro(residuals)[0]),
                'p_value': float(stats.shapiro(residuals)[1])
            }
        }
        
        lb_tests = {}
        for lag in [5, 10, 15, 20]:
            lb_test = acorr_ljungbox(residuals, lags=[lag], return_df=True)
            lb_tests[f'lag_{lag}'] = {
                'statistic': float(lb_test['lb_stat'].values[0]),
                'p_value': float(lb_test['lb_pvalue'].values[0]) 
            }
        
        squared_resid = residuals ** 2
        arch_tests = {}
        for lag in [5, 10, 15]:
            arch_test = acorr_ljungbox(squared_resid, lags=[lag], return_df=True)
            arch_tests[f'lag_{lag}'] = {
                'statistic': float(arch_test['lb_stat'].values[0]),
                'p_value': float(arch_test['lb_pvalue'].values[0])
            }
        
        acf_values = pd.Series(residuals).autocorr(lag=1)
        
        try:
            adf_test = adfuller(residuals)
            stationarity_test = {
                'adf_statistic': float(adf_test[0]),
                'p_value': float(adf_test[1]),
                'critical_values': {str(key): float(val) for key, val in adf_test[4].items()}
            }
        except:
            stationarity_test = None
        
        return {
            'model_name': model_name,
            'basic_stats': stats_info,
            'normality_tests': normality_tests,
            'ljung_box_tests': lb_tests,
            'arch_effect_tests': arch_tests,
            'first_order_autocorr': float(acf_values),
            'stationarity_test': stationarity_test
        }
        
    except:
        return None

def train_and_forecast_arima_garch(scaled_log_return_series_history, original_log_return_series, model_type, seq_len, pred_len, p_ar=1, q_ma=0, save_params=False):
    """
    ARIMA+GARCH模型训练和预测核心函数
    
    功能说明:
    --------
    1. 支持多种GARCH族模型的训练和预测
    2. 实现多步预测功能
    3. 自动模型参数估计和诊断
    4. 异常处理和ARMA模型回退机制
    
    支持的模型类型:
    -------------
    1. ARCH类: 
       - ARCH(1)_AR: AR均值 + ARCH(1)方差
       - ARCH(1)_Const: 常数均值 + ARCH(1)方差
    2. GARCH类:
       - GARCH(1,1)_AR: AR均值 + GARCH(1,1)方差
       - GARCH(1,1)_Const: 常数均值 + GARCH(1,1)方差
    3. EGARCH类:
       - EGARCH(1,1)_AR: AR均值 + EGARCH(1,1)方差 (非对称效应)
       - EGARCH(1,1)_Const: 常数均值 + EGARCH(1,1)方差
    4. GARCH-M类:
       - GARCH-M(1,1)_AR: AR均值-方差 + GARCH(1,1)
       - GARCH-M(1,1)_Const: 常数均值-方差 + GARCH(1,1)
    5. GJR-GARCH类:
       - GJR-GARCH(1,1)_AR: AR均值 + GJR-GARCH(1,1) (阈值GARCH)
    6. TGARCH类:
       - TGARCH(1,1)_AR: AR均值 + TGARCH(1,1) (阈值GARCH)
    
    参数:
    ----
    scaled_log_return_series_history : array-like
        标准化后的对数收益率历史序列 (用于ARMA回退)
    original_log_return_series : array-like
        原始对数收益率序列 (用于GARCH族模型，避免标准化影响)
    model_type : str
        模型类型标识符
    seq_len : int
        训练序列长度 (滚动窗口大小)
    pred_len : int
        预测步长 (1, 24, 30, 60, 90, 180天)
    p_ar : int, 默认=1
        AR(自回归)项阶数
    q_ma : int, 默认=0
        MA(移动平均)项阶数
    save_params : bool, 默认=False
        是否保存模型参数和诊断结果
        
    返回:
    ----
    tuple : (predictions, model_params, diagnostics_result)
        predictions : ndarray
            pred_len步的预测值
        model_params : dict or None
            模型参数字典 (当save_params=True时)
        diagnostics_result : dict or None
            模型诊断结果 (当save_params=True时)
            
    数学模型:
    --------
    GARCH(1,1)模型:
        均值方程: r_t = μ + φ₁r_{t-1} + εᵗ
        方差方程: σ²ᵗ = ω + α₁ε²_{t-1} + β₁σ²_{t-1}
        
    EGARCH(1,1)模型:
        ln(σ²ᵗ) = ω + α₁|ε_{t-1}|/σ_{t-1} + γ₁ε_{t-1}/σ_{t-1} + β₁ln(σ²_{t-1})
        
    GARCH-M模型:
        r_t = μ + λσᵗ + φ₁r_{t-1} + εᵗ (均值中包含条件方差项)
    """
    train_seq = original_log_return_series[-seq_len:]  # 使用原始序列
    garch_model_instance = None
    model_params = None
    diagnostics_result = None

    try:
        if 'GARCH-M' in model_type:
            if model_type == 'GARCH-M(1,1)_AR':
                base_model = arch_model(train_seq, mean='Zero', vol='GARCH', p=1, q=1)
                base_res = base_model.fit(disp='off', show_warning=False)
                conditional_vol = base_res.conditional_volatility
                idx = train_seq.index if hasattr(train_seq, 'index') else pd.RangeIndex(start=len(train_seq) - len(conditional_vol), stop=len(train_seq))
                x_for_arx = pd.DataFrame({'vol': conditional_vol}, index=idx[-len(conditional_vol):])
                
                arx_model = arch_model(train_seq, x=x_for_arx, mean='ARX', lags=p_ar, vol='Constant')
                arx_res = arx_model.fit(disp='off', show_warning=False)
                
                if save_params:
                    model_params = {
                        'mean_equation': arx_res.params.to_dict(),
                        'variance_equation': base_res.params.to_dict()
                    }
                    diagnostics_result = perform_model_diagnostics(arx_res.resid, model_type)
                
                vol_forecasts_obj = base_res.forecast(horizon=pred_len)
                predicted_variance = vol_forecasts_obj.variance.values[0, :pred_len]
                predicted_std_dev = np.sqrt(np.maximum(predicted_variance, 0))
                x_forecast_df = pd.DataFrame({'vol': predicted_std_dev}, index=np.arange(pred_len))
                mean_forecasts_obj = arx_res.forecast(horizon=pred_len, x=x_forecast_df)
                return mean_forecasts_obj.mean.values[0, :pred_len], model_params, diagnostics_result
            else:  
                try:
                    base_garch = arch_model(train_seq, mean='Constant', vol='GARCH', p=1, q=1)
                    base_res = base_garch.fit(disp='off', show_warning=False)
                    
                    if save_params:
                        model_params = {
                            'mean_equation': {'mu': float(base_res.params['mu'])},
                            'variance_equation': {k: float(v) for k, v in base_res.params.items() if k != 'mu'}
                        }
                        diagnostics_result = perform_model_diagnostics(base_res.resid, model_type)
                    
                    conditional_vol = base_res.conditional_volatility
                    const_param = base_res.params['mu']
                    
                    vol_forecasts = base_res.forecast(horizon=pred_len)
                    predicted_variance = vol_forecasts.variance.values[0, :pred_len]
                    predicted_std_dev = np.sqrt(np.maximum(predicted_variance, 0))
                    
                    risk_premium_coef = 0.1
                    garch_m_forecasts = const_param + risk_premium_coef * predicted_std_dev
                    return garch_m_forecasts, model_params, diagnostics_result
                except:
                    standard_garch = arch_model(train_seq, mean='Constant', vol='GARCH', p=1, q=1)
                    res = standard_garch.fit(disp='off', show_warning=False)
                    forecast_obj = res.forecast(horizon=pred_len)
                    return forecast_obj.mean.values[0, :pred_len], None, None
                
        else: 
            if model_type == 'ARCH(1)_AR':
                garch_model_instance = arch_model(train_seq, mean='AR', lags=p_ar, vol='ARCH', p=1, q=0)
            elif model_type == 'ARCH(1)_Const':
                garch_model_instance = arch_model(train_seq, mean='Constant', vol='ARCH', p=1, q=0)
            elif model_type == 'GARCH(1,1)_AR':
                garch_model_instance = arch_model(train_seq, mean='AR', lags=p_ar, vol='GARCH', p=1, o=0, q=1)
            elif model_type == 'GARCH(1,1)_Const':
                garch_model_instance = arch_model(train_seq, mean='Constant', vol='GARCH', p=1, o=0, q=1)
            elif model_type == 'EGARCH(1,1)_AR':
                garch_model_instance = arch_model(train_seq, mean='AR', lags=p_ar, vol='EGARCH', p=1, o=1, q=1)
            elif model_type == 'EGARCH(1,1)_Const':
                garch_model_instance = arch_model(train_seq, mean='Constant', vol='EGARCH', p=1, o=1, q=1)
            elif model_type == 'GJR-GARCH(1,1)_AR':
                garch_model_instance = arch_model(train_seq, mean='AR', lags=p_ar, vol='GARCH', p=1, o=1, q=1)
            elif model_type == 'GJR-GARCH(1,1)_Const':
                garch_model_instance = arch_model(train_seq, mean='Constant', vol='GARCH', p=1, o=1, q=1)
            elif model_type == 'TGARCH(1,1)_AR':
                garch_model_instance = arch_model(train_seq, mean='AR', lags=p_ar, vol='GARCH', p=1, o=1, q=1, power=1.0)
            elif model_type == 'TGARCH(1,1)_Const':
                garch_model_instance = arch_model(train_seq, mean='Constant', vol='GARCH', p=1, o=1, q=1, power=1.0)
            else:
                return np.zeros(pred_len), None, None
            
            try:
                fit_update_freq = 5 if 'EGARCH' in model_type or 'GJR-GARCH' in model_type or 'TGARCH' in model_type else 1
                results = garch_model_instance.fit(disp='off', show_warning=False, update_freq=fit_update_freq)
                
                if save_params:
                    mean_params = {}
                    variance_params = {}
                    
                    for param_name, param_value in results.params.items():
                        param_value_float = float(param_value)
                        if any(x in param_name.lower() for x in ['omega', 'alpha', 'beta', 'gamma']):
                            variance_params[param_name] = param_value_float
                        else:
                            mean_params[param_name] = param_value_float
                    
                    model_params = {
                        'mean_equation': mean_params,
                        'variance_equation': variance_params,
                        'ar_order': p_ar,
                        'ma_order': q_ma
                    }
                    
                    diagnostics_result = perform_model_diagnostics(results.resid, model_type)
                
                forecast_obj = results.forecast(horizon=pred_len)
                return forecast_obj.mean.values[0, :pred_len], model_params, diagnostics_result
                
            except:
                try:
                    arima_model = ARIMA(scaled_log_return_series_history[-seq_len:], order=(p_ar, 0, q_ma))
                    arima_results = arima_model.fit()
                    forecast_val = arima_results.forecast(steps=pred_len)
                    return np.array(forecast_val.values if isinstance(forecast_val, pd.Series) else forecast_val).flatten()[:pred_len], None, None
                except:
                    return np.zeros(pred_len), None, None
                    
    except:
        try:
            arima_model = ARIMA(scaled_log_return_series_history[-seq_len:], order=(p_ar, 0, q_ma))
            arima_results = arima_model.fit()
            forecast_val = arima_results.forecast(steps=pred_len)
            return np.array(forecast_val.values if isinstance(forecast_val, pd.Series) else forecast_val).flatten()[:pred_len], None, None
        except:
            return np.zeros(pred_len), None, None

    return forecast_obj.mean.values[0, :pred_len], model_params, diagnostics_result

def evaluate_model(y_true_prices, y_pred_prices):
    y_true_prices = np.asarray(y_true_prices).flatten()
    y_pred_prices = np.asarray(y_pred_prices).flatten()
    
    if len(y_true_prices) != len(y_pred_prices):
        min_length = min(len(y_true_prices), len(y_pred_prices))
        y_true_prices = y_true_prices[:min_length]
        y_pred_prices = y_pred_prices[:min_length]
    
    if len(y_true_prices) == 0 or len(y_pred_prices) == 0:
        return {'MSE': float('inf'), 'RMSE': float('inf'), 'MAE': float('inf'), 'MAX_AE': float('inf'), 'MAPE': float('inf'), 'R2': -float('inf')}
    
    if not np.all(np.isfinite(y_pred_prices)):
        finite_preds = y_pred_prices[np.isfinite(y_pred_prices)]
        mean_finite_pred = np.mean(finite_preds) if len(finite_preds) > 0 else (np.mean(y_true_prices) if len(y_true_prices) > 0 else 0)
        y_pred_prices = np.nan_to_num(y_pred_prices, nan=mean_finite_pred, posinf=1e12, neginf=-1e12)

    mse = mean_squared_error(y_true_prices, y_pred_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_prices, y_pred_prices)
    max_ae = np.max(np.abs(y_true_prices - y_pred_prices))
    r2 = r2_score(y_true_prices, y_pred_prices)
    
    mask = np.abs(y_true_prices) > 1e-9
    mape = np.mean(np.abs((y_true_prices[mask] - y_pred_prices[mask]) / y_true_prices[mask])) * 100 if np.sum(mask) > 0 else np.nan
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAX_AE': max_ae, 'MAPE': mape, 'R2': r2}

def run_naive_baseline_forecast(data_df, original_price_col_name, pred_len, step_size, test_set_ratio=0.15):
    """
    朴素基线预测函数 - 修改为动态测试集版本
    
    修改说明:
    --------
    1. 从固定150点测试集改为按比例(15%)动态计算测试集大小
    2. 确保预测窗口不超出数据边界
    3. 真实值和预测值数组长度严格匹配
    4. 正确处理多步预测的边界情况
    """
    original_prices = data_df[original_price_col_name].values
    num_total_points = len(original_prices)

    # 计算动态测试集大小
    dynamic_test_set_size = int(num_total_points * test_set_ratio)
    print(f"动态测试集大小: {dynamic_test_set_size} (总数据点: {num_total_points}, 测试集比例: {test_set_ratio:.1%})")

    if num_total_points <= dynamic_test_set_size:
        print(f"警告: 数据点总数 ({num_total_points}) 不足以支持 {dynamic_test_set_size} 点的测试集。")
        return np.array([]), np.array([])

    test_start_idx = num_total_points - dynamic_test_set_size
    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []

    # 修复: 确保循环范围不会导致越界
    # 最后一个有效的预测起始位置应该是 num_total_points - pred_len
    max_start_idx = min(num_total_points - pred_len, num_total_points - 1)
    
    for i in tqdm(range(test_start_idx, max_start_idx + 1, step_size), 
                 desc=f"Naive Baseline 预测进度 ({pred_len}天)", file=sys.stderr):
        if i < 1 : continue 
        
        # 修复: 确保不会超出数据边界
        end_idx = min(i + pred_len, num_total_points)
        actual_price_levels_this_window = original_prices[i : end_idx] 
        
        # 修复: 只有当实际数据长度等于pred_len时才处理
        if len(actual_price_levels_this_window) != pred_len:
            continue
            
        value_from_previous_day = original_prices[i-1] 
        predicted_price_levels_this_window = [value_from_previous_day] * pred_len
        
        all_true_price_levels_collected.append(actual_price_levels_this_window)
        all_predicted_price_levels_collected.append(predicted_price_levels_this_window)

    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([])

    # 修复: 确保连接后的数组长度一致
    true_values = np.concatenate(all_true_price_levels_collected)
    pred_values = np.concatenate(all_predicted_price_levels_collected)
    
    # 调试信息: 打印数组长度以便检查
    print(f"朴素基线预测 - 真实值长度: {len(true_values)}, 预测值长度: {len(pred_values)}")
    
    # 修复: 如果仍然长度不匹配，截取到较短的长度
    min_length = min(len(true_values), len(pred_values))
    return true_values[:min_length], pred_values[:min_length]

def rolling_forecast(data_df, original_price_col_name, log_return_col_name, 
                     model_type, seq_len, pred_len, step_size=1, p=1, q=0, test_set_ratio=0.15):
    """
    滚动窗口多步预测函数 - 修改为动态测试集版本
    
    功能说明:
    --------
    1. 实现滚动窗口的多步时间序列预测
    2. 支持1到180天的预测步长
    3. 从对数收益率预测重构到价格水平
    4. 实时异常检测和处理
    5. 全面的残差分析和LB检验
    
    修改说明:
    --------
    1. 从固定150点测试集改为按比例(15%)动态计算测试集大小
    2. 其他预测逻辑保持不变
    
    预测流程:
    --------
    1. 数据标准化: 对对数收益率进行Z-score标准化
    2. 滚动窗口: 固定窗口大小，逐步滑动预测
    3. 模型训练: 在每个窗口上训练GARCH模型
    4. 多步预测: 递归/直接预测未来pred_len步
    5. 价格重构: 从对数收益率预测重构实际价格
    6. 残差收集: 收集每个预测窗口的残差用于诊断
    
    参数:
    ----
    data_df : pd.DataFrame
        包含汇率和对数收益率的数据框
    original_price_col_name : str
        原始价格列名 (如 'Close')
    log_return_col_name : str
        对数收益率列名
    model_type : str
        GARCH模型类型
    seq_len : int
        滚动窗口长度 (历史数据长度)
    pred_len : int
        预测步长 (1, 24, 30, 60, 90, 180天)
    step_size : int, 默认=1
        滚动步长
    p : int, 默认=1
        AR项阶数
    q : int, 默认=0
        MA项阶数
    test_set_ratio : float, 默认=0.15
        测试集比例 (15%)
        
    返回:
    ----
    tuple : (true_values, predictions, model_info, lb_test_result)
        true_values : ndarray
            真实价格值 (展平的所有预测窗口)
        predictions : ndarray
            预测价格值 (展平的所有预测窗口)
        model_info : dict
            第一个窗口的模型信息和参数
        lb_test_result : dict
            对所有预测残差的Ljung-Box检验结果
            
    数学原理:
    --------
    价格重构公式: P_t = P_{t-1} * exp(r_t)
    其中 P_t 为t时刻价格，r_t 为t时刻对数收益率预测值
    
    多步预测通过递归应用: P_{t+k} = P_{t+k-1} * exp(r_{t+k})
    """
    log_returns = data_df[log_return_col_name].values
    original_prices = data_df[original_price_col_name].values 

    log_return_scaler = StandardScaler()
    scaled_log_returns = log_return_scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
    
    num_total_log_points = len(scaled_log_returns)
    
    # 计算动态测试集大小
    dynamic_test_set_size = int(num_total_log_points * test_set_ratio)
    test_start_idx_log = num_total_log_points - dynamic_test_set_size
    print(f"动态测试集大小: {dynamic_test_set_size} (总数据点: {num_total_log_points}, 测试集比例: {test_set_ratio:.1%})")
    
    if test_start_idx_log < 0: 
        print(f"警告: 模型 {model_type} 计算的 test_start_idx_log ({test_start_idx_log}) < 0。没有足够的测试数据。")
        return np.array([]), np.array([]), None, None

    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []
    first_window_model_info = None
    all_price_level_residuals = []  # 收集汇率尺度的残差（每个窗口只取第一步预测的残差）
    
    extreme_value_threshold = 10.0

    for i in tqdm(range(test_start_idx_log, num_total_log_points - pred_len + 1, step_size),
                 desc=f"{model_type} 预测进度 ({pred_len}天)", file=sys.stderr):
        
        current_history_for_model_input = scaled_log_returns[:i]
        original_log_returns_for_model = log_returns[:i]  
        train_seq_actually_used_by_model = current_history_for_model_input[-seq_len:]

        last_actual_price_for_reconstruction = original_prices[i-1] 

        save_params = (i == test_start_idx_log)
        predicted_std_log_returns_raw, model_params, diagnostics_result = train_and_forecast_arima_garch(
            current_history_for_model_input, original_log_returns_for_model, model_type, seq_len, pred_len, 
            p_ar=p, q_ma=q, save_params=save_params
        )

        if save_params:
            first_window_model_info = {
                'model_type': model_type,
                'parameters': model_params,
                'diagnostics': diagnostics_result,
                'ar_order': p,
                'ma_order': q
            }

        use_arma_fallback_in_rolling = False
        
        if not isinstance(predicted_std_log_returns_raw, np.ndarray) or \
           predicted_std_log_returns_raw.shape != (pred_len,) or \
           np.any(np.isnan(predicted_std_log_returns_raw)) or \
           np.any(np.isinf(predicted_std_log_returns_raw)) or \
           np.any(np.abs(predicted_std_log_returns_raw) > extreme_value_threshold):
            use_arma_fallback_in_rolling = True

        if use_arma_fallback_in_rolling:
            predicted_std_log_returns, _, _ = train_and_forecast_pure_arma(
                current_history_for_model_input, seq_len, pred_len, p_ar=p, q_ma=q
            )
            if not (isinstance(predicted_std_log_returns, np.ndarray) and \
                    predicted_std_log_returns.shape == (pred_len,) and \
                    not np.any(np.isnan(predicted_std_log_returns)) and \
                    not np.any(np.isinf(predicted_std_log_returns)) and \
                    not np.any(np.abs(predicted_std_log_returns) > extreme_value_threshold)):
                predicted_std_log_returns = np.zeros(pred_len)
        else:
            predicted_std_log_returns = predicted_std_log_returns_raw
        
        predicted_log_returns_for_window = log_return_scaler.inverse_transform(
            np.array(predicted_std_log_returns).reshape(-1, 1)
        ).flatten()

        reconstructed_price_forecasts_this_window = []
        current_reconstructed_price = last_actual_price_for_reconstruction
        for log_ret_pred_step in predicted_log_returns_for_window:
            log_ret_pred_step = np.clip(log_ret_pred_step, -5, 5) 
            current_reconstructed_price = current_reconstructed_price * np.exp(log_ret_pred_step)
            reconstructed_price_forecasts_this_window.append(current_reconstructed_price)
        
        end_idx = min(i + pred_len, len(original_prices))
        actual_price_levels_this_window = original_prices[i : end_idx]
        
        if len(actual_price_levels_this_window) == pred_len and len(reconstructed_price_forecasts_this_window) == pred_len:
            all_predicted_price_levels_collected.append(reconstructed_price_forecasts_this_window)
            all_true_price_levels_collected.append(actual_price_levels_this_window)
            
            first_step_residual = reconstructed_price_forecasts_this_window[0] - actual_price_levels_this_window[0]
            all_price_level_residuals.append(first_step_residual)

    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([]), None, None

    # 对汇率尺度的残差进行完整的LB检验
    if len(all_price_level_residuals) > 0:
        residuals_array = np.array(all_price_level_residuals)
        lb_tests = {}
        for lag in [5, 10, 15, 20]:
            lb_test = acorr_ljungbox(residuals_array, lags=[lag], return_df=True)
            lb_tests[f'lag_{lag}'] = {
                'statistic': float(lb_test['lb_stat'].values[0]),
                'p_value': float(lb_test['lb_pvalue'].values[0])
            }
        
        lb_test_result = {
            'ljung_box_tests': lb_tests,
            'residuals_summary': {
                'mean': float(np.mean(residuals_array)),
                'std': float(np.std(residuals_array)),
                'n_samples': len(residuals_array),
                'skewness': float(stats.skew(residuals_array)),
                'kurtosis': float(stats.kurtosis(residuals_array, fisher=True) + 3)
            }
        }
    else:
        lb_test_result = None

    # 修复: 连接数组前确保长度一致性
    true_values = np.concatenate(all_true_price_levels_collected)
    pred_values = np.concatenate(all_predicted_price_levels_collected)
    
    # 修复: 最终安全检查，确保长度一致
    min_length = min(len(true_values), len(pred_values))
    if len(true_values) != len(pred_values):
        print(f"警告: 最终数组长度不一致，截取到长度: {min_length}")
        true_values = true_values[:min_length]
        pred_values = pred_values[:min_length]

    return true_values, pred_values, first_window_model_info, lb_test_result

def train_and_forecast_pure_arma(scaled_log_return_series_history, seq_len, pred_len, p_ar=1, q_ma=0, save_params=False):
    train_seq = scaled_log_return_series_history[-seq_len:]
    model_params = None
    lb_test_result = None
    
    arima_model = ARIMA(train_seq, order=(p_ar, 0, q_ma))
    try:
        arima_results = arima_model.fit()
        
        if save_params:
            model_params = {
                'mean_equation': {
                    'ar': arima_results.arparams.tolist() if hasattr(arima_results, 'arparams') else [],
                    'ma': arima_results.maparams.tolist() if hasattr(arima_results, 'maparams') else [],
                    'const': float(arima_results.params[0]) if len(arima_results.params) > 0 else 0.0
                }
            }
            residuals = arima_results.resid
            lb_tests = {}
            for lag in [5, 10, 15, 20]:
                lb_test = acorr_ljungbox(residuals, lags=[lag], return_df=True)
                lb_tests[f'lag_{lag}'] = {
                    'statistic': float(lb_test['lb_stat'].values[0]),
                    'p_value': float(lb_test['lb_pvalue'].values[0])
                }
            
            lb_test_result = {
                'ljung_box_tests': lb_tests,
                'residuals_summary': {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'n_samples': len(residuals),
                    'skewness': float(stats.skew(residuals)),
                    'kurtosis': float(stats.kurtosis(residuals, fisher=True) + 3)
                }
            }
        
        forecast_val = arima_results.forecast(steps=pred_len)
        if isinstance(forecast_val, pd.Series):
            forecast_val = forecast_val.values
        return np.array(forecast_val).flatten()[:pred_len], model_params, lb_test_result
    except:
        return np.zeros(pred_len), None, None

def rolling_forecast_pure_arma(data_df, original_price_col_name, log_return_col_name, 
                     seq_len, pred_len, step_size=1, p=1, q=0, test_set_ratio=0.15):
    log_returns = data_df[log_return_col_name].values
    original_prices = data_df[original_price_col_name].values

    log_return_scaler = StandardScaler()
    scaled_log_returns = log_return_scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
    
    num_total_log_points = len(scaled_log_returns)
    
    # 计算动态测试集大小
    dynamic_test_set_size = int(num_total_log_points * test_set_ratio)
    test_start_idx_log = num_total_log_points - dynamic_test_set_size
    
    if test_start_idx_log < 0:
        print(f"警告: 纯ARMA模型计算的 test_start_idx_log ({test_start_idx_log}) < 0。没有足够的测试数据。")
        return np.array([]), np.array([]), None, None

    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []
    first_window_model_info = None
    all_price_level_residuals = []  # 收集汇率尺度的残差（每个窗口只取第一步预测的残差）
    
    extreme_value_threshold = 10.0 # 同 GARCH rolling forecast 一致的阈值

    for i in tqdm(range(test_start_idx_log, num_total_log_points - pred_len + 1, step_size),
                 desc=f"Pure ARMA({p},{q}) 预测进度 ({pred_len}天)", file=sys.stderr):
        
        current_std_log_ret_history = scaled_log_returns[:i]
        last_actual_price_for_reconstruction = original_prices[i-1]

        # 只在第一个窗口保存模型参数
        save_params = (i == test_start_idx_log)
        predicted_std_log_returns, model_params, lb_test_result_unused = train_and_forecast_pure_arma(
            current_std_log_ret_history, seq_len, pred_len, p_ar=p, q_ma=q, save_params=save_params
        )

        if save_params and model_params is not None:
            first_window_model_info = {
                'model_type': f'Pure ARMA({p},{q})',
                'parameters': model_params,
                'lb_test': lb_test_result_unused
            }
        
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
        
        # 修复: 确保实际价格窗口不超出数据边界
        end_idx = min(i + pred_len, len(original_prices))
        actual_price_levels_this_window = original_prices[i : end_idx]
        
        # 修复: 只有当实际数据长度等于pred_len时才添加到结果中
        if len(actual_price_levels_this_window) == pred_len and len(reconstructed_price_forecasts_this_window) == pred_len:
            all_predicted_price_levels_collected.append(reconstructed_price_forecasts_this_window)
            all_true_price_levels_collected.append(actual_price_levels_this_window)
            
            # 只收集每个窗口的第一步预测残差
            first_step_residual = reconstructed_price_forecasts_this_window[0] - actual_price_levels_this_window[0]
            all_price_level_residuals.append(first_step_residual)
        else:
            print(f"警告: 跳过长度不匹配的窗口 - 实际数据长度: {len(actual_price_levels_this_window)}, 预测长度: {len(reconstructed_price_forecasts_this_window)}, 要求长度: {pred_len}")

    if not all_predicted_price_levels_collected:
        return np.array([]), np.array([]), None, None

    # 对汇率尺度的残差进行完整的LB检验
    if len(all_price_level_residuals) > 0:
        residuals_array = np.array(all_price_level_residuals)
        lb_tests = {}
        for lag in [5, 10, 15, 20]:
            lb_test = acorr_ljungbox(residuals_array, lags=[lag], return_df=True)
            lb_tests[f'lag_{lag}'] = {
                'statistic': float(lb_test['lb_stat'].values[0]),
                'p_value': float(lb_test['lb_pvalue'].values[0])
            }
        
        lb_test_result = {
            'ljung_box_tests': lb_tests,
            'residuals_summary': {
                'mean': float(np.mean(residuals_array)),
                'std': float(np.std(residuals_array)),
                'n_samples': len(residuals_array),
                'skewness': float(stats.skew(residuals_array)),
                'kurtosis': float(stats.kurtosis(residuals_array, fisher=True) + 3)
            }
        }
    else:
        lb_test_result = None

    # 修复: 连接数组前确保长度一致性
    true_values = np.concatenate(all_true_price_levels_collected)
    pred_values = np.concatenate(all_predicted_price_levels_collected)
    
    # 修复: 最终安全检查，确保长度一致
    min_length = min(len(true_values), len(pred_values))
    if len(true_values) != len(pred_values):
        print(f"警告: 最终数组长度不一致，截取到长度: {min_length}")
        true_values = true_values[:min_length]
        pred_values = pred_values[:min_length]

    return true_values, pred_values, first_window_model_info, lb_test_result

def main(args):
    log_return_col = 'log_return_gbp_cny' 
    test_set_ratio = 0.15  

    try:
        data_df = load_and_prepare_data(args.root_path, args.data_path, args.target, log_return_col)
    except ValueError as e:
        print(f"数据加载或预处理失败: {e}")
        return

    num_total_points = len(data_df)
    dynamic_test_set_size = int(num_total_points * test_set_ratio)
    
    if num_total_points <= dynamic_test_set_size:
        return
        
    num_train_points_for_first_window = num_total_points - dynamic_test_set_size
    
    if num_train_points_for_first_window < args.seq_len:
        return

    if not data_df[log_return_col].empty:
        perform_stationarity_test(data_df[log_return_col], series_name=f"{args.target} 的对数收益率 (GBP/CNY)")
    else:
        return

    base_garch_architectures = ['ARCH', 'GARCH', 'EGARCH', 'GARCH-M', 'GJR-GARCH', 'TGARCH']
    multistep_unstable_architectures = ['EGARCH', 'TGARCH']
    
    mean_types = ['AR', 'Const']
    
    def generate_model_configs(architectures, mean_types, arma_params):
        configs = []
        
        for p, q in arma_params:
            configs.append({
                'name': f'Pure ARMA({p},{q})',
                'type': 'Pure ARMA',
                'p': p, 'q': q,
                'architecture': None,
                'mean': None
            })
        
        for arch in architectures:
            for mean in mean_types:
                for p, q in arma_params:
                    model_name = f'{arch}(1,1)_{mean}_p{p}q{q}' if arch != 'ARCH' else f'{arch}(1)_{mean}_p{p}q{q}'
                    configs.append({
                        'name': model_name,
                        'type': 'GARCH',
                        'p': p, 'q': q,
                        'architecture': arch,
                        'mean': mean
                    })
        
        return configs
    
    pred_lens = [1, 24, 30, 60, 90, 180]
    arma_params = [(1, 0), (1, 1), (2, 0)]
    seq_lens_to_test = [31]
    
    best_overall_metrics = {'MSE': float('inf'), 'model': None, 'p': None, 'q': None, 'seq_len': None, 'metrics': None}
    all_configs_comparison = {}
    
    for current_seq_len in seq_lens_to_test:
        args.seq_len = current_seq_len
        print(f"\n{'#'*80}")
        print(f"测试滚动窗口大小 (seq_len): {current_seq_len}")
        print(f"{'#'*80}")
    
        if num_train_points_for_first_window < current_seq_len:
            continue
            
        for pred_len in pred_lens:
            print(f"\n{'*'*80}")
            print(f"当前预测步长: {pred_len}天 (seq_len={current_seq_len})")
            print(f"{'*'*80}")

            if pred_len == 1:
                architectures_to_use = base_garch_architectures
            else:
                architectures_to_use = [arch for arch in base_garch_architectures 
                                      if arch not in multistep_unstable_architectures]
            
            model_configs = generate_model_configs(architectures_to_use, mean_types, arma_params)
            
            print(f"\n{'='*80}")
            print(f"运行完整模型集 (seq_len={current_seq_len}, pred_len={pred_len})")
            print(f"{'='*80}")
            
            results_dir = f'arima_garch_results_logret_GBPCNY_seq{current_seq_len}_{pred_len}step_last{int(test_set_ratio*100)}pct_test'
            os.makedirs(results_dir, exist_ok=True)
            
            run_results = {}
            metrics_dict = {}
            
            print(f"\n{'-'*50}\n模型: Naive Baseline (PrevDay) - {pred_len}步预测\n{'-'*50}")
            start_time_naive = time.time()
            naive_actuals, naive_preds = run_naive_baseline_forecast(
                data_df, args.target, pred_len, args.step_size, test_set_ratio
            )
            elapsed_time_naive = time.time() - start_time_naive

            if len(naive_actuals) > 0:
                eval_metrics_naive = evaluate_model(naive_actuals, naive_preds)
                run_results["Naive Baseline (PrevDay)"] = {
                    'metrics': eval_metrics_naive, 'true_values': naive_actuals,
                    'predictions': naive_preds, 'time': elapsed_time_naive,
                    'lb_test': None
                }
                metrics_dict["Naive Baseline (PrevDay)"] = eval_metrics_naive
                print(f"执行时间: {elapsed_time_naive:.2f}秒")
                for name, val_metric in eval_metrics_naive.items():
                    print(f"{name}: {val_metric:.8f}{'%' if name == 'MAPE' else ''}")

            for config in model_configs:
                model_name = config['name']
                model_type = config['type']
                p, q = config['p'], config['q']
                architecture = config['architecture']
                mean = config['mean']
                
                print(f"\n{'-'*50}\n模型: {model_name} - {pred_len}步预测\n{'-'*50}")
                start_time = time.time()
                
                if model_type == 'Pure ARMA':
                    actuals, preds, first_window_model_info, lb_test_result = rolling_forecast_pure_arma(
                        data_df, args.target, log_return_col, current_seq_len, pred_len, 
                        args.step_size, p, q, test_set_ratio=test_set_ratio
                    )
                else:
                    if architecture == 'ARCH':
                        legacy_model_type = f'{architecture}(1)_{mean}'
                    else:
                        legacy_model_type = f'{architecture}(1,1)_{mean}'
                    
                    actuals, preds, first_window_model_info, lb_test_result = rolling_forecast(
                        data_df, args.target, log_return_col, legacy_model_type,
                        current_seq_len, pred_len, args.step_size, p=p, q=q, test_set_ratio=test_set_ratio
                    )
                
                elapsed_time = time.time() - start_time
                
                if len(actuals) > 0:
                    eval_metrics = evaluate_model(actuals, preds)
                    run_results[model_name] = {
                        'metrics': eval_metrics, 'true_values': actuals,
                        'predictions': preds, 'time': elapsed_time,
                        'lb_test': lb_test_result, 'first_window_info': first_window_model_info
                    }
                    metrics_dict[model_name] = eval_metrics
                    print(f"执行时间: {elapsed_time:.2f}秒")
                    for name, val_metric in eval_metrics.items():
                        print(f"{name}: {val_metric:.8f}{'%' if name == 'MAPE' else ''}")
                    
                    if lb_test_result and 'ljung_box_tests' in lb_test_result:
                        print(f"\nLjung-Box检验结果:")
                        for lag, test_result in lb_test_result['ljung_box_tests'].items():
                            print(f"  {lag}: 统计量={test_result['statistic']:.4f}, p值={test_result['p_value']:.4f}")

            if run_results:
                model_params_file = os.path.join(results_dir, f'model_params_GBPCNY_seq{current_seq_len}_{pred_len}step.json')
                model_params_dict = {}
                for model_name, result_data in run_results.items():
                    if 'first_window_info' in result_data and result_data['first_window_info'] is not None:
                        model_params_dict[model_name] = result_data['first_window_info']
                
                if model_params_dict:
                    with open(model_params_file, 'w', encoding='utf-8') as f:
                        json.dump(model_params_dict, f, indent=4, ensure_ascii=False)
                
                plot_results(run_results, pred_len, results_dir, f"seq{current_seq_len}_{pred_len}step")
                
                with open(os.path.join(results_dir, f'results_GBPCNY_seq{current_seq_len}_pred_len_{pred_len}.pkl'), 'wb') as f:
                    pickle.dump(run_results, f)
                
                summary_input_for_table = {pred_len: run_results}
                generate_summary_table(summary_input_for_table, results_dir, f"seq{current_seq_len}_{pred_len}step")

            config_key = f"seq{current_seq_len}_{pred_len}step"
            if metrics_dict:
                all_configs_comparison[config_key] = {
                    'best_model_name': min(metrics_dict, key=lambda k: metrics_dict[k]['MSE']),
                    'best_mse': min(m['MSE'] for m in metrics_dict.values()),
                    'all_metrics_in_config': metrics_dict,
                    'pred_len': pred_len
                }
                
                best_model_in_this_config = min(metrics_dict, key=lambda k: metrics_dict[k]['MSE'])
                best_mse_in_this_config = metrics_dict[best_model_in_this_config]['MSE']
                
                if best_mse_in_this_config < best_overall_metrics['MSE']:
                    p_extracted = None
                    q_extracted = None
                    if 'p' in model_configs[0]:
                        for config in model_configs:
                            if config['name'] == best_model_in_this_config:
                                p_extracted = config['p']
                                q_extracted = config['q']
                                break
                    
                    best_overall_metrics.update({
                        'MSE': best_mse_in_this_config,
                        'model': f'{best_model_in_this_config} - {pred_len}步',
                        'p': p_extracted,
                        'q': q_extracted,
                        'seq_len': current_seq_len,
                        'pred_len': pred_len,
                        'metrics': metrics_dict[best_model_in_this_config]
                    })

    print("\n" + "="*100)
    print("多步预测实验完成 - 所有配置和预测步长的总结 (GBP/CNY):")
    print("="*100)
    
    results_by_pred_len = {}
    for config_label, summary_res in all_configs_comparison.items():
        pred_len = summary_res.get('pred_len', 1)
        if pred_len not in results_by_pred_len:
            results_by_pred_len[pred_len] = []
        results_by_pred_len[pred_len].append((config_label, summary_res))
    
    for pred_len in sorted(results_by_pred_len.keys()):
        print(f"\n{'-'*80}")
        print(f"预测步长: {pred_len}天")
        print(f"{'-'*80}")
        
        configs_for_this_pred_len = results_by_pred_len[pred_len]
        configs_for_this_pred_len.sort(key=lambda x: x[1]['best_mse'])
        
        for rank, (config_label, summary_res) in enumerate(configs_for_this_pred_len, 1):
            print(f"  {rank}. 配置: {config_label}")
            print(f"     最佳模型: {summary_res['best_model_name']}")
            print(f"     最低MSE: {summary_res['best_mse']:.8f}")
            print()
    
    print("\n" + "="*100)
    print("全局最佳模型配置 (所有预测步长中MSE最低的模型):")
    print("="*100)
    if best_overall_metrics['metrics']:
        print(f"模型类型: {best_overall_metrics['model']}")
        if best_overall_metrics['p'] is not None:
            print(f"ARMA参数: p={best_overall_metrics['p']}, q={best_overall_metrics['q']}")
        print(f"滚动窗口大小: {best_overall_metrics['seq_len']}")
        print(f"预测步长: {best_overall_metrics.get('pred_len', 1)}天")
        print(f"\n评估指标:")
        for metric_name, value in best_overall_metrics['metrics'].items():
            print(f"  {metric_name}: {value:.8f}{'%' if metric_name == 'MAPE' else ''}")
    else:
        print("未能找到最佳模型。")
    
    print("\n" + "="*100)
    print("实验说明 (GBP/CNY版本):")
    print("="*100)
    print("1. 模型类型:")
    print("   - Naive Baseline: 使用前一天的值作为所有未来预测")
    print("   - Pure ARMA: 纯时间序列自回归移动平均模型")
    print("   - GARCH族模型: 包括ARCH、GARCH、EGARCH、GARCH-M等异方差模型")
    print("2. 预测步长: 1天、24天、30天、60天、90天、180天的多步预测")
    print("3. 评估指标: MSE、RMSE、MAE、MAX_AE、MAPE、R²")
    print("4. 测试设置: 数据集的最后15%作为动态测试集")
    print("5. 诊断检验: Ljung-Box检验用于残差序列相关性检验")
    print("6. 数据来源: 英镑兑人民币汇率 (GBP/CNY)")
    print("="*100)



def plot_results(results_dict, pred_len, results_dir_path, config_label):
    """
    绘制预测结果并保存图像，包含置信区间和多步预测效果的可视化
    修复：限制图像尺寸以避免过大的图像文件
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
    
    # 修复：限制图像高度，避免过大尺寸
    # 每个模型最多4英寸高度，总高度不超过32英寸
    max_height_per_model = 4
    max_total_height = 32
    
    # 如果模型太多，分批绘制
    models_per_batch = min(num_models_to_plot, max(1, max_total_height // max_height_per_model))
    
    for batch_idx in range(0, num_models_to_plot, models_per_batch):
        batch_models = sorted_model_keys[batch_idx:batch_idx + models_per_batch]
        batch_size = len(batch_models)
        
        # 为每个批次创建图像
        plt.figure(figsize=(20, max_height_per_model * batch_size)) 
    
        for batch_plot_idx, model_type_key in enumerate(batch_models):
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
            plt.subplot(batch_size, 2, 2*batch_plot_idx + 1)
            
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
            
            # 局部预测视图
            plt.subplot(batch_size, 2, 2*batch_plot_idx + 2)
            
            # 选择一个代表性的预测窗口进行展示
            window_start = len(true_vals) // 2  # 从中间位置开始
            local_window_size = pred_len * 3  # 显示3个预测周期
            
            # 绘制局部预测窗口的置信区间
            plt.fill_between(range(window_start, window_start + local_window_size),
                            lower_bound[window_start:window_start + local_window_size],
                            upper_bound[window_start:window_start + local_window_size],
                            color='gray', alpha=0.2, label='95% 置信区间')
            
            # 绘制局部实际值和预测值
            plt.plot(range(window_start, window_start + local_window_size),
                    true_vals[window_start:window_start + local_window_size],
                    label='实际价格', color='blue', marker='o')
            plt.plot(range(window_start, window_start + local_window_size),
                    pred_vals[window_start:window_start + local_window_size],
                    label='预测价格', color='red', linestyle='--', marker='x')
            
            # 标记预测步长
            for i in range(window_start, window_start + local_window_size - pred_len + 1, pred_len):
                plt.axvspan(i, i + pred_len, color='yellow', alpha=0.1)
                plt.axvline(x=i, color='green', linestyle=':', alpha=0.5)
                if i == window_start:
                    plt.annotate(f'预测步长: {pred_len}天',
                               xy=(i, plt.ylim()[0]),
                               xytext=(i, plt.ylim()[0] - (plt.ylim()[1] - plt.ylim()[0])*0.1),
                               arrowprops=dict(facecolor='black', shrink=0.05),
                               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
            
            plt.title(f'{model_type_key} ({config_label})\n局部预测效果展示', fontsize=12)
            plt.xlabel('时间步长', fontsize=10)
            plt.ylabel('汇率', fontsize=10)
            plt.legend(fontsize=10, loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加说明文字
            plt.figtext(0.02, 0.98 - (batch_plot_idx)/batch_size, 
                       f"说明:\n"
                       f"1. 蓝线(实线+圆点): 实际价格\n"
                       f"2. 红线(虚线+叉号): 预测价格\n"
                       f"3. 灰色区域: 95% 置信区间\n"
                       f"4. 黄色区域: {pred_len}天预测窗口\n"
                       f"5. 绿色虚线: 预测起始点",
                       fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 为每个批次保存单独的图像文件
        batch_suffix = f"_batch{batch_idx//models_per_batch + 1}" if num_models_to_plot > models_per_batch else ""
        plt.savefig(os.path.join(results_dir_path, f'plot_price_level_{config_label}_pred_len_{pred_len}{batch_suffix}.png'), 
                    dpi=100, bbox_inches='tight')  # 进一步降低DPI以减小文件大小
        plt.close()

def generate_summary_table(all_results_summary, results_dir_path, arima_params_label):
    summary_data = {'Model': []}
    metric_names = ['MSE', 'RMSE', 'MAE', 'MAX_AE', 'MAPE', 'R2', 'Time(s)']  # 确保包含MAX_AE
    
    pred_lengths_present = sorted(all_results_summary.keys())
    if not pred_lengths_present: return pd.DataFrame()

    for pred_len_val in pred_lengths_present:
        for metric_n in metric_names:
            summary_data[f'{metric_n}_{pred_len_val}'] = []
    
    model_types_present_set = set()
    for pred_len_val in pred_lengths_present:
        if isinstance(all_results_summary[pred_len_val], dict):
            model_types_present_set.update(all_results_summary[pred_len_val].keys())
    
    ordered_model_types = []
    if "Naive Baseline (PrevDay)" in model_types_present_set: ordered_model_types.append("Naive Baseline (PrevDay)")
    
    pure_arma_keys = sorted([k for k in model_types_present_set if "Pure ARMA" in k])
    ordered_model_types.extend(pure_arma_keys)

    other_garch_models = sorted([k for k in model_types_present_set if k not in ordered_model_types])
    ordered_model_types.extend(other_garch_models)

    if not ordered_model_types: return pd.DataFrame()

    for model_t in ordered_model_types:
        summary_data['Model'].append(model_t)
        for pred_len_val in pred_lengths_present:
            model_result_data = all_results_summary.get(pred_len_val, {}).get(model_t)
            if model_result_data and 'metrics' in model_result_data and 'time' in model_result_data:
                for metric_n in metric_names:
                    val = model_result_data['time'] if metric_n == 'Time(s)' else model_result_data['metrics'].get(metric_n, float('nan'))
                    
                    if metric_n == 'Time(s)': 
                        formatted_val = f"{val:.2f}"  # 执行时间保持2位小数
                    else:
                        formatted_val = f"{val:.8f}{'%' if metric_n == 'MAPE' else ''}"  # 所有评估指标统一8位小数
                    summary_data[f'{metric_n}_{pred_len_val}'].append(formatted_val)
            else:
                for metric_n in metric_names:
                    summary_data[f'{metric_n}_{pred_len_val}'].append('N/A')
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(results_dir_path, f'summary_table_{arima_params_label}.csv'), index=False)
    print(f"\n结果汇总 ({arima_params_label} - 动态15%测试集, 预测长度 {pred_lengths_present[0]}天):")
    pd.set_option('display.float_format', lambda x: '%.8f' if isinstance(x, float) else str(x))  # 设置pandas显示格式为8位小数
    print(df_summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARIMA+GARCH汇率预测 (GBP/CNY, 对数收益率, 多步预测, 动态15%测试集)')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='数据根目录')
    parser.add_argument('--data_path', type=str, default='英镑兑人民币_short_series.csv', help='数据文件路径')
    parser.add_argument('--target', type=str, default='Close', help='原始汇率目标变量列名')
    parser.add_argument('--seq_len', type=int, default=31, help='ARIMA+GARCH历史对数收益率长度')
    parser.add_argument('--step_size', type=int, default=1, help='滚动窗口步长')
    
    args = parser.parse_args()
    main(args) 