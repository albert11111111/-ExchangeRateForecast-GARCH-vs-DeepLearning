#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
比较ARIMA+GARCH基准模型与TimesNet模型的预测结果
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json

def load_arima_garch_results(results_dir):
    """
    加载ARIMA+GARCH模型的结果
    
    参数:
    - results_dir: 结果目录
    
    返回:
    - 结果字典
    """
    all_results_file = os.path.join(results_dir, 'all_results.pkl')
    
    if not os.path.exists(all_results_file):
        raise FileNotFoundError(f"未找到ARIMA+GARCH结果文件: {all_results_file}")
    
    with open(all_results_file, 'rb') as f:
        all_results = pickle.load(f)
    
    return all_results

def load_timesnet_results(checkpoints_dir, pred_lens):
    """
    加载TimesNet模型的结果
    
    参数:
    - checkpoints_dir: 模型检查点目录
    - pred_lens: 预测长度列表
    
    返回:
    - 结果字典
    """
    timesnet_results = {}
    
    for pred_len in pred_lens:
        # 寻找对应预测长度的模型结果
        model_dir = f"GBP_CNY_96_{pred_len}"
        model_path = os.path.join(checkpoints_dir, model_dir)
        
        if not os.path.exists(model_path):
            print(f"警告: 未找到TimesNet在预测长度{pred_len}的结果目录: {model_path}")
            continue
        
        # 读取指标文件
        metrics_file = os.path.join(model_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            timesnet_results[pred_len] = metrics
        else:
            print(f"警告: 未找到TimesNet在预测长度{pred_len}的指标文件: {metrics_file}")
    
    return timesnet_results

def create_comparison_table(arima_garch_results, timesnet_results, metrics=['MSE', 'RMSE', 'MAE', 'MAPE']):
    """
    创建比较表格
    
    参数:
    - arima_garch_results: ARIMA+GARCH结果
    - timesnet_results: TimesNet结果
    - metrics: 要比较的指标列表
    
    返回:
    - 比较表格DataFrame
    """
    # 初始化结果表格
    comparison = {'Prediction Length': [], 'Model': []}
    for metric in metrics:
        comparison[metric] = []
    
    # 预测长度列表
    pred_lens = sorted(list(set(list(arima_garch_results.keys()) + list(timesnet_results.keys()))))
    
    # 确定最佳ARIMA+GARCH模型
    best_arima_garch_model = {}
    for pred_len, results in arima_garch_results.items():
        best_model = None
        best_rmse = float('inf')
        
        for model_type, result in results.items():
            if result['metrics']['RMSE'] < best_rmse:
                best_rmse = result['metrics']['RMSE']
                best_model = model_type
        
        best_arima_garch_model[pred_len] = best_model
    
    # 填充结果表格
    for pred_len in pred_lens:
        # 添加ARIMA+GARCH结果
        if pred_len in arima_garch_results:
            for model_type, result in arima_garch_results[pred_len].items():
                # 标记最佳模型
                model_name = f"{model_type}"
                if model_type == best_arima_garch_model[pred_len]:
                    model_name += " (Best)"
                
                comparison['Prediction Length'].append(pred_len)
                comparison['Model'].append(model_name)
                
                for metric in metrics:
                    if metric in result['metrics']:
                        comparison[metric].append(result['metrics'][metric])
                    else:
                        comparison[metric].append(np.nan)
        
        # 添加TimesNet结果
        if pred_len in timesnet_results:
            comparison['Prediction Length'].append(pred_len)
            comparison['Model'].append('TimesNet')
            
            for metric in metrics:
                # 检查JSON中的指标名称是否匹配
                metric_name = metric
                if metric.lower() in timesnet_results[pred_len]:
                    metric_name = metric.lower()
                
                if metric_name in timesnet_results[pred_len]:
                    comparison[metric].append(timesnet_results[pred_len][metric_name])
                else:
                    comparison[metric].append(np.nan)
    
    # 创建DataFrame
    df_comparison = pd.DataFrame(comparison)
    
    return df_comparison

def plot_comparison(df_comparison, metrics=['RMSE', 'MAE'], save_dir='.'):
    """
    绘制比较图表
    
    参数:
    - df_comparison: 比较表格DataFrame
    - metrics: 要绘制的指标列表
    - save_dir: 保存目录
    """
    # 设置样式
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))
    
    # 对每个指标绘制子图
    for i, metric in enumerate(metrics, 1):
        plt.subplot(len(metrics), 1, i)
        
        # 转换数据格式为绘图用
        plot_data = []
        for _, row in df_comparison.iterrows():
            pred_len = row['Prediction Length']
            model = row['Model']
            value = row[metric]
            
            if not pd.isna(value):
                plot_data.append({
                    'Prediction Length': pred_len,
                    'Model': model,
                    metric: value
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # 绘制柱状图
        ax = sns.barplot(x='Prediction Length', y=metric, hue='Model', data=plot_df)
        
        # 添加数值标签
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.4f}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom',
                       xytext=(0, 5),
                       textcoords='offset points')
        
        plt.title(f'比较不同模型的{metric}指标')
        plt.ylabel(metric)
        plt.xlabel('预测长度（天）')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    plt.close()

def main(args):
    """
    主函数
    """
    # 确认输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载ARIMA+GARCH结果
    print("加载ARIMA+GARCH结果...")
    arima_garch_results = load_arima_garch_results(args.arima_garch_dir)
    
    # 确定预测长度列表
    pred_lens = sorted(list(arima_garch_results.keys()))
    print(f"找到预测长度: {pred_lens}")
    
    # 加载TimesNet结果
    print("加载TimesNet结果...")
    timesnet_results = load_timesnet_results(args.timesnet_dir, pred_lens)
    
    # 创建比较表格
    print("创建比较表格...")
    df_comparison = create_comparison_table(arima_garch_results, timesnet_results)
    
    # 保存比较表格
    comparison_file = os.path.join(args.output_dir, 'model_comparison.csv')
    df_comparison.to_csv(comparison_file, index=False)
    print(f"比较表格已保存至: {comparison_file}")
    
    # 打印比较表格
    print("\n比较结果:")
    print(df_comparison.to_string())
    
    # 绘制比较图表
    print("绘制比较图表...")
    plot_comparison(df_comparison, save_dir=args.output_dir)
    print(f"比较图表已保存至: {os.path.join(args.output_dir, 'model_comparison.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='比较ARIMA+GARCH与TimesNet模型的预测结果')
    
    parser.add_argument('--arima_garch_dir', type=str, default='arima_garch_results',
                       help='ARIMA+GARCH结果目录')
    parser.add_argument('--timesnet_dir', type=str, default='./checkpoints',
                       help='TimesNet检查点目录')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    main(args) 