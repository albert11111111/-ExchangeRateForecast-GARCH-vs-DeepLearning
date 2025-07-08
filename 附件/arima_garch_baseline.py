#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller # 新增导入
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    df.reset_index(drop=True)
    
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


def train_and_forecast_arima_garch(scaled_log_return_series_history, model_type, seq_len, pred_len, p=1, q=0):
    """
    基于（标准化的）对数收益率历史训练ARIMA+GARCH模型并预测未来pred_len步的（标准化的）对数收益率。
    """
    train_seq = scaled_log_return_series_history[-seq_len:] # 使用历史的最后seq_len部分
    
    arima_model = ARIMA(train_seq, order=(p, 0, q))
    try:
        arima_results = arima_model.fit()
    except Exception as e_arima:
        print(f"警告: ARIMA({p},0,{q}) 模型拟合失败: {e_arima}。返回零预测。")
        return np.zeros(pred_len)

    residuals = arima_results.resid
    
    if model_type == 'ARCH(1)':
        garch_model = arch_model(residuals, vol='ARCH', p=1, q=0)
    elif model_type == 'GARCH(1,1)':
        garch_model = arch_model(residuals, vol='GARCH', p=1, q=1)
    elif model_type == 'EGARCH(1,1)':
        try:
            # 明确指定 o=1, 并尝试使用 dist='t'
            garch_model = arch_model(residuals, vol='EGARCH', p=1, o=1, q=1, dist='t')
            print("尝试使用 EGARCH(p=1, o=1, q=1, dist='t') 模型。")
        except Exception as e_garch_create_egarch_t:
            print(f"警告: EGARCH(p=1, o=1, q=1, dist='t') 模型创建失败 ({e_garch_create_egarch_t})，尝试 EGARCH(p=1, o=1, q=1) with normal dist。")
            try:
                garch_model = arch_model(residuals, vol='EGARCH', p=1, o=1, q=1)
                print("尝试使用 EGARCH(p=1, o=1, q=1, dist='normal') 模型。")
            except Exception as e_garch_create_egarch_norm:
                print(f"警告: EGARCH(p=1, o=1, q=1, dist='normal') 模型创建也失败 ({e_garch_create_egarch_norm})，尝试使用GARCH(1,1)替代。")
                garch_model = arch_model(residuals, vol='GARCH', p=1, q=1)
                model_type = 'GARCH(1,1)' # Update model_type to reflect fallback
                print("回退到 GARCH(1,1) 模型。")
    elif model_type == 'GARCH-M(1,1)':
        garch_model = arch_model(residuals, vol='GARCH', p=1, q=1, mean='ARX', lags=1) # GARCH-M通常对收益率建模
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    try:
        # 增加maxiter并尝试抑制部分拟合警告 - 暂时移除 show_warning=False 以观察
        garch_results = garch_model.fit(disp='off', options={'maxiter': 300}) # Removed show_warning=False
        
        # ARIMA预测（均值部分，即预测的对数收益率的均值部分）
        arima_mean_forecast_std_log_ret = arima_results.forecast(steps=pred_len)
        if hasattr(arima_mean_forecast_std_log_ret, 'values'):
             arima_mean_forecast_std_log_ret = arima_mean_forecast_std_log_ret.values
        
        # GARCH波动率预测并结合均值
        if model_type == 'ARCH(1)':
            forecasts_std_log_ret = np.zeros(pred_len)
            last_resid = residuals[-1] if len(residuals)>0 else 0
            omega = garch_results.params['omega']
            alpha = garch_results.params['alpha[1]']
            for i in range(pred_len):
                sigma2 = omega + alpha * last_resid**2
                sigma = np.sqrt(max(1e-12, sigma2)) # 确保正值
                eps = np.random.normal(0, 1)
                forecasts_std_log_ret[i] = arima_mean_forecast_std_log_ret[i] + sigma * eps
                last_resid = sigma * eps
            return forecasts_std_log_ret
            
        elif model_type == 'GARCH(1,1)':
            forecasts_std_log_ret = np.zeros(pred_len)
            last_resid = residuals[-1] if len(residuals)>0 else 0
            last_var = garch_results.conditional_volatility[-1]**2 if len(garch_results.conditional_volatility)>0 else np.mean(residuals**2)
            
            omega = garch_results.params['omega']
            alpha_param = garch_results.params.get('alpha[1]', 0.0) # Handle missing alpha if GARCH(0,1) like
            beta_param = garch_results.params.get('beta[1]', 0.0)

            for i in range(pred_len):
                sigma2 = omega + alpha_param * last_resid**2 + beta_param * last_var
                sigma = np.sqrt(max(1e-12, sigma2))
                eps = np.random.normal(0, 1)
                forecasts_std_log_ret[i] = arima_mean_forecast_std_log_ret[i] + sigma * eps
                last_resid = sigma * eps
                last_var = sigma2
            return forecasts_std_log_ret
            
        elif model_type == 'EGARCH(1,1)':
            try:
                egarch_vol_forecast = garch_results.forecast(horizon=pred_len, method='simulation', reindex=False)
                forecasted_conditional_variances = egarch_vol_forecast.variance.iloc[0].values
                
                # 更稳健地处理NaN和极端值
                # 使用输入残差的方差作为NaN的替代，如果残差为空则用1.0
                nan_replacement_variance = np.var(residuals) if residuals.size > 0 else 1.0
                if nan_replacement_variance <= 0: # Ensure non-negative, non-zero replacement
                    nan_replacement_variance = 1e-6

                forecasted_conditional_variances = np.nan_to_num(
                    forecasted_conditional_variances, 
                    nan=nan_replacement_variance, 
                    posinf=nan_replacement_variance * 100, # Cap large positive infinities relative to data
                    neginf=1e-6 # Variances cannot be negative, replace with small positive
                )
                # 对标准化的对数收益率的方差，上限100可能是一个起点 (原为1e12)
                # 下限确保为正
                forecasted_conditional_variances = np.clip(forecasted_conditional_variances, 1e-9, 100.0)

                forecasts_std_log_ret = np.zeros(pred_len)
                current_arima_mean = arima_mean_forecast_std_log_ret[:pred_len]
                for i in range(pred_len):
                    sigma_val_sq = forecasted_conditional_variances[i] if i < len(forecasted_conditional_variances) else forecasted_conditional_variances[-1]
                    sigma = np.sqrt(sigma_val_sq) # Already clipped
                    eps = np.random.normal(0, 1)
                    mean_val = current_arima_mean[i] if i < len(current_arima_mean) else current_arima_mean[-1]
                    forecasts_std_log_ret[i] = mean_val + sigma * eps
                return forecasts_std_log_ret
            except Exception as e_egarch_pred:
                print(f"警告: EGARCH ({model_type}) 模型波动率预测失败 ({str(e_egarch_pred)})，仅使用ARIMA均值预测（标准化的对数收益率）。")
                return arima_mean_forecast_std_log_ret[:pred_len]
            
        elif model_type == 'GARCH-M(1,1)':
            # For GARCH-M, the mean forecast from ARIMA already incorporates the volatility term implicitly.
            # However, the GARCH part is for the residuals of this mean.
            # The forecast from arima_results.forecast() is for the mean E[r_t | F_{t-1}].
            # We still need to simulate the volatility for the stochastic part.
            # This is similar to GARCH(1,1) but arima_mean is already the full mean.
            forecasts_std_log_ret = np.zeros(pred_len)
            last_resid = residuals[-1] if len(residuals)>0 else 0
            last_var = garch_results.conditional_volatility[-1]**2 if len(garch_results.conditional_volatility)>0 else np.mean(residuals**2)
            
            omega = garch_results.params['omega']
            alpha_param = garch_results.params.get('alpha[1]', 0.0)
            beta_param = garch_results.params.get('beta[1]', 0.0)

            for i in range(pred_len):
                sigma2 = omega + alpha_param * last_resid**2 + beta_param * last_var
                sigma = np.sqrt(max(1e-12, sigma2))
                eps = np.random.normal(0, 1)
                # The ARIMA mean forecast IS the E[log_return], so we add sigma*eps
                forecasts_std_log_ret[i] = arima_mean_forecast_std_log_ret[i] + sigma * eps 
                last_resid = sigma * eps # This might need refinement based on GARCH-M definition.
                                         # For simplicity, assume residual for GARCH part is sigma*eps
                last_var = sigma2
            return forecasts_std_log_ret

    except Exception as e_garch_fit:
        print(f"警告: {model_type} GARCH部分处理失败 ({str(e_garch_fit)})，仅使用ARIMA均值预测（标准化的对数收益率）。")
        arima_fc = arima_results.forecast(steps=pred_len)
        return arima_fc.values if hasattr(arima_fc, 'values') else arima_fc


def evaluate_model(y_true_prices, y_pred_prices):
    """
    评估模型性能 (y_true_prices 和 y_pred_prices 必须是原始价格水平)
    """
    y_true_prices = np.asarray(y_true_prices).flatten()
    y_pred_prices = np.asarray(y_pred_prices).flatten()

    print(f"    用于评估的真实价格 (前5个): {y_true_prices[:5]}")
    print(f"    用于评估的预测价格 (前5个): {y_pred_prices[:5]}")
    if len(y_true_prices) > 5:
        print(f"    用于评估的真实价格 (后5个): {y_true_prices[-5:]}")
        print(f"    用于评估的预测价格 (后5个): {y_pred_prices[-5:]}")
    
    # 处理预测中可能出现的 NaN 或 Inf
    if not np.all(np.isfinite(y_pred_prices)):
        print("警告: 预测价格包含非有限值 (NaN/inf)。正在进行清理...")
        # Replace NaNs with mean of finite predictions, Infs with a large number
        finite_preds = y_pred_prices[np.isfinite(y_pred_prices)]
        mean_finite_pred = np.mean(finite_preds) if len(finite_preds) > 0 else (np.mean(y_true_prices) if len(y_true_prices) > 0 else 0) # Fallback
        
        y_pred_prices = np.nan_to_num(y_pred_prices, nan=mean_finite_pred, posinf=1e12, neginf=-1e12) # Cap large values

    mse = mean_squared_error(y_true_prices, y_pred_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_prices, y_pred_prices)
    
    mask = np.abs(y_true_prices) > 1e-9 # Avoid division by zero for MAPE
    if np.sum(mask) == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true_prices[mask] - y_pred_prices[mask]) / y_true_prices[mask])) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def rolling_forecast(data_df, original_price_col_name, log_return_col_name, 
                     model_type, seq_len, pred_len, step_size=1, p=1, q=0):
    """
    使用滚动窗口进行预测。模型在对数收益率上训练，预测对数收益率，
    然后将预测的对数收益率还原为价格水平。
    返回原始价格水平的真实值和预测值。
    """
    log_returns = data_df[log_return_col_name].values
    original_prices = data_df[original_price_col_name].values

    log_return_scaler = StandardScaler()
    scaled_log_returns = log_return_scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()
    
    all_true_price_levels_collected = []
    all_predicted_price_levels_collected = []
    
    # 确定测试集起始位置（基于对数收益率序列的长度）
    # train_val_split_ratio = 0.85 (70% train + 15% val, so 15% test)
    # test_start_idx is the index of the first log_return to be part of a test window's actuals
    # This also means it's the first log_return to be predicted
    num_total_points = len(scaled_log_returns)
    test_start_idx = int(num_total_points * 0.85) 

    # The loop variable 'i' will be the index in 'scaled_log_returns' (and 'original_prices')
    # marking the START of the 'pred_len' period to be predicted.
    # History for ARIMA is data *before* index 'i'.
    # last_actual_price for reconstruction is original_prices[i-1].
    
    min_history_needed_for_arima_garch = seq_len 
    if test_start_idx < min_history_needed_for_arima_garch:
        print(f"警告: test_start_idx ({test_start_idx}) 小于模型所需最小历史 ({min_history_needed_for_arima_garch}). 可能没有足够的初始数据点进行滚动预测。")
        # Adjust test_start_idx or raise error. For now, let tqdm handle empty range if it occurs.
        test_start_idx = min_history_needed_for_arima_garch # Ensure at least seq_len history

    # Ensure the loop doesn't go out of bounds for actual values
    # The last possible 'i' is such that 'i + pred_len -1' is the last index of original_prices/scaled_log_returns
    end_loop_idx = num_total_points - pred_len 

    for i in tqdm(range(test_start_idx, end_loop_idx + 1, step_size),
                 desc=f"{model_type} 预测进度 ({pred_len}天)", file=sys.stderr):
        
        # History for ARIMA+GARCH (standardized log returns up to point i-1)
        current_std_log_ret_history = scaled_log_returns[:i]
        if len(current_std_log_ret_history) < seq_len: # Should not happen if test_start_idx is set correctly
            continue 

        # Get the last actual price *before* the prediction period starts (original_prices[i-1])
        last_actual_price_for_reconstruction = original_prices[i-1]

        # Predict next 'pred_len' steps of *standardized log returns*
        predicted_std_log_returns = train_and_forecast_arima_garch(
            current_std_log_ret_history, model_type, seq_len, pred_len, p, q
        )
        
        # Inverse transform predicted standardized log returns to actual log returns
        predicted_log_returns_for_window = log_return_scaler.inverse_transform(
            predicted_std_log_returns.reshape(-1, 1)
        ).flatten()

        # Reconstruct price level forecasts
        reconstructed_price_forecasts_this_window = []
        current_reconstructed_price = last_actual_price_for_reconstruction
        for log_ret_pred_step in predicted_log_returns_for_window:
            # Clip log_ret_pred_step to avoid exp overflow if it's excessively large
            log_ret_pred_step = np.clip(log_ret_pred_step, -5, 5) # Limits return to e^-5 to e^5 per step
            current_reconstructed_price = current_reconstructed_price * np.exp(log_ret_pred_step)
            reconstructed_price_forecasts_this_window.append(current_reconstructed_price)
        
        # Get actual price levels for comparison for this window
        actual_price_levels_this_window = original_prices[i : i + pred_len]
        
        all_predicted_price_levels_collected.append(reconstructed_price_forecasts_this_window)
        all_true_price_levels_collected.append(actual_price_levels_this_window)

    if not all_predicted_price_levels_collected:
        print(f"警告: 模型 {model_type} 未生成任何预测结果。")
        return np.array([]), np.array([]), None 

    final_true_prices = np.concatenate(all_true_price_levels_collected)
    final_predicted_prices = np.concatenate(all_predicted_price_levels_collected)
    
    return final_true_prices, final_predicted_prices, None # Return None for scaler, as evaluation is on prices


def main(args):
    """
    主函数
    """
    log_return_col = 'log_return' # Define a name for the log return column

    # 1. 加载并准备数据（计算对数收益率）
    try:
        data_df = load_and_prepare_data(args.root_path, args.data_path, args.target, log_return_col)
    except ValueError as e:
        print(f"数据加载或预处理失败: {e}")
        return

    # 2. 对对数收益率序列进行平稳性检验
    if not data_df[log_return_col].empty:
        perform_stationarity_test(data_df[log_return_col], series_name=f"{args.target} 的对数收益率")
    else:
        print("对数收益率序列为空，无法进行平稳性检验。")
        return

    model_types = [
        'ARCH(1)', 'GARCH(1,1)', 'EGARCH(1,1)', 'GARCH-M(1,1)'
    ]
    pred_lens = [7, 14, 30, 60, 90, 180] # 修改：加入7天和14天预测
    
    # 修改：文件名和目录名中包含ARIMA的p,q参数
    arima_params_str = f"p{args.p}q{args.q}"
    results_dir = f'arima_garch_results_logret_model_{arima_params_str}'
    os.makedirs(results_dir, exist_ok=True)
    all_run_results = {}

    for pred_len_steps in pred_lens:
        print(f"\n{'='*50}")
        print(f"预测长度: {pred_len_steps}天")
        print(f"{'='*50}")
        
        current_pred_len_results = {}
        for model_name in model_types:
            print(f"\n{'-'*50}")
            print(f"模型: {model_name} (ARIMA p={args.p}, q={args.q})") # 明确打印p,q
            print(f"{'-'*50}")
            
            start_time_model = time.time()
            
            # 使用滚动窗口进行预测（基于对数收益率，返回价格水平）
            actual_prices, predicted_prices, _ = rolling_forecast(
                data_df, 
                args.target, # original_price_col_name
                log_return_col, # log_return_col_name for modeling
                model_name, 
                args.seq_len, 
                pred_len_steps,
                args.step_size,
                args.p, # Pass p from args
                args.q  # Pass q from args
            )
            
            elapsed_time_model = time.time() - start_time_model
            
            if len(actual_prices) > 0 and len(predicted_prices) > 0:
                # 评估模型（基于价格水平）
                eval_metrics = evaluate_model(actual_prices, predicted_prices) # No scaler needed
                
                current_pred_len_results[model_name] = {
                    'metrics': eval_metrics,
                    # Store actuals and predictions (already price levels)
                    # These are concatenated 1D arrays from all rolling windows
                    'true_values': actual_prices, 
                    'predictions': predicted_prices,
                    'time': elapsed_time_model
                }
                
                print(f"执行时间: {elapsed_time_model:.2f}秒")
                for metric_name, metric_val in eval_metrics.items():
                    print(f"{metric_name}: {metric_val:.4f}")
                
                # 保存NPY和CSV文件 (数据已经是原始价格维度)
                num_rolling_windows_run = len(actual_prices) // pred_len_steps
                model_name_clean = model_name.replace("(", "").replace(")", "").replace(",", "_")
                
                if num_rolling_windows_run > 0 and len(actual_prices) % pred_len_steps == 0:
                    # Reshape for NPY: (num_rolling_windows, pred_len, 1)
                    true_prices_npy = actual_prices.reshape(num_rolling_windows_run, pred_len_steps, 1)
                    predicted_prices_npy = predicted_prices.reshape(num_rolling_windows_run, pred_len_steps, 1)

                    # 修改：文件名包含p,q
                    npy_true_fname = os.path.join(results_dir, f'true_prices_{arima_params_str}_{model_name_clean}_len{pred_len_steps}.npy')
                    npy_pred_fname = os.path.join(results_dir, f'pred_prices_{arima_params_str}_{model_name_clean}_len{pred_len_steps}.npy')
                    np.save(npy_true_fname, true_prices_npy)
                    np.save(npy_pred_fname, predicted_prices_npy)
                    print(f"已保存价格水平真实值NPY: {npy_true_fname}")
                    print(f"已保存价格水平预测值NPY: {npy_pred_fname}")

                    # Reshape for CSV: (num_rolling_windows, pred_len)
                    true_prices_csv = actual_prices.reshape(num_rolling_windows_run, pred_len_steps)
                    predicted_prices_csv = predicted_prices.reshape(num_rolling_windows_run, pred_len_steps)
                    csv_cols = [f'pred_step_{j+1}' for j in range(pred_len_steps)]
                    df_true_prices_csv = pd.DataFrame(true_prices_csv, columns=csv_cols)
                    df_pred_prices_csv = pd.DataFrame(predicted_prices_csv, columns=csv_cols)
                    
                    # 修改：文件名包含p,q
                    csv_true_fname = os.path.join(results_dir, f'true_prices_{arima_params_str}_{model_name_clean}_len{pred_len_steps}.csv')
                    csv_pred_fname = os.path.join(results_dir, f'pred_prices_{arima_params_str}_{model_name_clean}_len{pred_len_steps}.csv')
                    df_true_prices_csv.to_csv(csv_true_fname, index=False)
                    df_pred_prices_csv.to_csv(csv_pred_fname, index=False)
                    print(f"已保存价格水平真实值CSV: {csv_true_fname}")
                    print(f"已保存价格水平预测值CSV: {csv_pred_fname}")
                elif len(actual_prices) > 0:
                    print(f"警告: 结果长度 {len(actual_prices)} 不能被 pred_len {pred_len_steps} 整除。跳过NPY和CSV保存。")
            else:
                print("没有足够的数据进行评估或预测失败。")
        
        all_run_results[pred_len_steps] = current_pred_len_results
        
        # 修改：文件名包含p,q
        results_pkl_file = os.path.join(results_dir, f'results_logret_model_{arima_params_str}_pred_len_{pred_len_steps}.pkl')
        with open(results_pkl_file, 'wb') as f:
            pickle.dump(current_pred_len_results, f)
        
        # 绘制结果图 (基于价格水平)
        if current_pred_len_results: # Check if there are results to plot
            plot_results(current_pred_len_results, pred_len_steps, results_dir, arima_params_str) # Pass arima_params_str for filename
    
    # 修改：文件名包含p,q
    all_results_pkl_file = os.path.join(results_dir, f'all_results_logret_model_{arima_params_str}.pkl')
    with open(all_results_pkl_file, 'wb') as f:
        pickle.dump(all_run_results, f)
    
    if all_run_results:
        generate_summary_table(all_run_results, results_dir, arima_params_str) # Pass arima_params_str for filename

def plot_results(results_dict, pred_len, results_dir_path, arima_params_label):
    """
    绘制预测结果图 (数据已经是原始价格水平)
    """
    num_models_to_plot = len(results_dict)
    if num_models_to_plot == 0:
        print("没有模型结果可供绘图。")
        return

    plt.figure(figsize=(15, 5 * num_models_to_plot)) # Adjusted height
    
    plot_idx = 1
    for model_type_key, result_data in results_dict.items():
        if 'true_values' not in result_data or 'predictions' not in result_data:
            print(f"跳过绘图 {model_type_key}，缺少 true_values 或 predictions。")
            continue

        true_values_price_level = np.asarray(result_data['true_values']).flatten()
        predicted_values_price_level = np.asarray(result_data['predictions']).flatten()
        
        if len(true_values_price_level) == 0 or len(predicted_values_price_level) == 0:
            print(f"跳过绘图 {model_type_key}，真实值或预测值为空。")
            continue
            
        plt.subplot(num_models_to_plot, 1, plot_idx)
        # Plot only a portion if the series are too long for clarity
        max_points_to_plot = 1000 
        plt.plot(true_values_price_level[:max_points_to_plot], label='实际价格', color='blue')
        plt.plot(predicted_values_price_level[:max_points_to_plot], label='预测价格', color='red', linestyle='--')
        # 修改：图标题包含p,q
        plt.title(f'{model_type_key} ({arima_params_label}) - 预测长度 {pred_len}天 (价格水平)')
        plt.xlabel('时间步长 (聚合后的测试窗口)')
        plt.ylabel('汇率')
        plt.legend()
        plt.grid(True)
        plot_idx += 1
    
    if plot_idx > 1: # Only save if at least one plot was made
        plt.tight_layout()
        # 修改：文件名包含p,q
        plot_filename = os.path.join(results_dir_path, f'plot_price_level_{arima_params_label}_pred_len_{pred_len}.png')
        plt.savefig(plot_filename)
        print(f"已保存价格水平绘图: {plot_filename}")
    else:
        print(f"没有为预测长度 {pred_len} 生成任何绘图。")
    plt.close()

def generate_summary_table(all_results_summary, results_dir_path, arima_params_label):
    """
    生成结果汇总表 (基于价格水平的指标)
    """
    summary_data = {'Model': []}
    metric_names = ['MSE', 'RMSE', 'MAE', 'MAPE', 'Time(s)']
    
    # Determine all prediction lengths present
    pred_lengths_present = sorted(all_results_summary.keys())
    if not pred_lengths_present:
        print("没有结果可用于生成汇总表。")
        return pd.DataFrame()

    for pred_len_val in pred_lengths_present:
        for metric_n in metric_names:
            summary_data[f'{metric_n}_{pred_len_val}'] = []
    
    model_types_present = []
    for pred_len_val in pred_lengths_present:
        if isinstance(all_results_summary[pred_len_val], dict):
            model_types_present.extend(list(all_results_summary[pred_len_val].keys()))
    model_types_present = sorted(list(set(model_types_present))) # Unique sorted model types

    if not model_types_present:
        print("在结果中未找到模型类型。")
        return pd.DataFrame()

    for model_t in model_types_present:
        summary_data['Model'].append(model_t)
        for pred_len_val in pred_lengths_present:
            pred_len_results_dict = all_results_summary.get(pred_len_val, {})
            model_result = pred_len_results_dict.get(model_t)
            
            if model_result and 'metrics' in model_result and 'time' in model_result:
                for metric_n in metric_names:
                    if metric_n == 'Time(s)':
                        summary_data[f'{metric_n}_{pred_len_val}'].append(f"{model_result['time']:.2f}")
                    elif metric_n in model_result['metrics']:
                        summary_data[f'{metric_n}_{pred_len_val}'].append(f"{model_result['metrics'][metric_n]:.4f}")
                    else:
                        summary_data[f'{metric_n}_{pred_len_val}'].append('N/A')
            else:
                for metric_n in metric_names:
                    summary_data[f'{metric_n}_{pred_len_val}'].append('N/A')
    
    df_summary_final = pd.DataFrame(summary_data)
    
    # 修改：文件名包含p,q
    summary_csv_path = os.path.join(results_dir_path, f'summary_table_logret_model_{arima_params_label}.csv')
    df_summary_final.to_csv(summary_csv_path, index=False)
    
    print(f"\n结果汇总 ({arima_params_label} - 基于对数收益率建模，价格水平评估):")
    print(df_summary_final.to_string())
    
    return df_summary_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARIMA+GARCH汇率预测基准模型 (基于对数收益率)')
    
    parser.add_argument('--root_path', type=str, default='./dataset/', help='数据根目录')
    parser.add_argument('--data_path', type=str, default='英镑兑人民币_20250324_102930.csv', help='数据文件路径')
    parser.add_argument('--target', type=str, default='rate', help='原始汇率目标变量列名')
    
    parser.add_argument('--p', type=int, default=1, help='ARIMA模型AR项的阶数 (应用于对数收益率)') # Default p changed
    parser.add_argument('--q', type=int, default=0, help='ARIMA模型MA项的阶数 (应用于对数收益率)')
    parser.add_argument('--seq_len', type=int, default=96, help='用于训练ARIMA+GARCH的历史对数收益率长度')
    parser.add_argument('--step_size', type=int, default=1, help='滚动窗口的步长')
    
    args = parser.parse_args()
    
    main(args) 