#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import sys
import pandas as pd
import numpy as np

def load_pkl_file(pkl_file_path):
    """
    加载pickle格式的模型结果文件
    
    参数:
    - pkl_file_path: pickle文件路径
    
    返回:
    - 加载的pickle数据
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"成功加载文件: {pkl_file_path}")
        return data
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

def print_structure(data, max_depth=2, prefix=''):
    """
    打印数据结构的层次和类型
    
    参数:
    - data: 要分析的数据
    - max_depth: 最大递归深度
    - prefix: 打印的前缀
    """
    if max_depth <= 0:
        print(f"{prefix}...(达到最大深度)")
        return
    
    if isinstance(data, dict):
        print(f"{prefix}字典，包含 {len(data)} 个键:")
        for i, (key, value) in enumerate(data.items()):
            if i < 10:  # 限制打印数量
                print(f"{prefix}  键 '{key}':")
                print_structure(value, max_depth-1, prefix+'    ')
            else:
                print(f"{prefix}  ...剩余 {len(data)-10} 个键")
                break
    elif isinstance(data, list):
        print(f"{prefix}列表，包含 {len(data)} 个元素:")
        if len(data) > 0:
            if len(data) <= 5:
                for i, item in enumerate(data):
                    print(f"{prefix}  元素 {i}:")
                    print_structure(item, max_depth-1, prefix+'    ')
            else:
                print(f"{prefix}  显示前 3 个元素:")
                for i in range(3):
                    print(f"{prefix}  元素 {i}:")
                    print_structure(data[i], max_depth-1, prefix+'    ')
                print(f"{prefix}  ...共 {len(data)} 个元素")
    elif isinstance(data, np.ndarray):
        print(f"{prefix}NumPy数组，形状 {data.shape}，类型 {data.dtype}")
        if data.size <= 5:
            print(f"{prefix}  值: {data}")
    elif isinstance(data, pd.DataFrame):
        print(f"{prefix}DataFrame，形状 {data.shape}")
        print(f"{prefix}  列: {list(data.columns)}")
        if data.shape[0] <= 5:
            print(f"{prefix}  前几行:\n{data.head()}")
        else:
            print(f"{prefix}  前3行:\n{data.head(3)}")
    else:
        print(f"{prefix}类型: {type(data).__name__}")
        try:
            if hasattr(data, '__len__') and len(data) <= 100:
                print(f"{prefix}  值: {data}")
            else:
                print(f"{prefix}  值: {str(data)[:100]} ...")
        except:
            print(f"{prefix}  无法打印值")

def display_metrics(data):
    """
    显示模型评估指标
    
    参数:
    - data: 包含模型结果的字典
    """
    for model_key, model_data in data.items():
        if 'metrics' in model_data:
            print(f"\n模型: {model_key}")
            metrics = model_data['metrics']
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")

def extract_model_names(data):
    """
    提取所有模型名称
    
    参数:
    - data: 包含模型结果的字典
    
    返回:
    - 模型名称列表
    """
    return list(data.keys())

def extract_predicted_values(data, model_key):
    """
    提取特定模型的预测值
    
    参数:
    - data: 包含模型结果的字典
    - model_key: 模型键名
    
    返回:
    - 预测值数组
    """
    if model_key in data and 'predictions' in data[model_key]:
        return data[model_key]['predictions']
    return None

def extract_true_values(data, model_key):
    """
    提取特定模型的真实值
    
    参数:
    - data: 包含模型结果的字典
    - model_key: 模型键名
    
    返回:
    - 真实值数组
    """
    if model_key in data and 'true_values' in data[model_key]:
        return data[model_key]['true_values']
    return None

def main():
    if len(sys.argv) < 2:
        print("用法: python view_pkl_results.py <pkl文件路径>")
        print("例如: python view_pkl_results.py ./arima_garch_results_logret_USDJPY_optimized_seq96_multistep_ratio_70_15_15/all_results_USDJPY_seq96_multistep.pkl")
        
        # 自动搜索当前目录下的pkl文件
        pkl_files = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.pkl'):
                    pkl_files.append(os.path.join(root, file))
        
        if pkl_files:
            print("\n在当前目录及子目录中找到以下pkl文件:")
            for i, file in enumerate(pkl_files):
                print(f"{i+1}. {file}")
            print("\n您可以选择上述文件之一进行查看")
        return
    
    pkl_file_path = sys.argv[1]
    data = load_pkl_file(pkl_file_path)
    
    if data is None:
        return
    
    print("\n数据结构概览:")
    print_structure(data)
    
    model_names = extract_model_names(data)
    print(f"\n包含的模型 ({len(model_names)}):")
    for i, name in enumerate(model_names):
        print(f"{i+1}. {name}")
    
    # 交互式查看特定模型的详细信息
    while True:
        user_input = input("\n输入模型编号查看详细信息 (输入 'q' 退出): ")
        if user_input.lower() == 'q':
            break
        
        try:
            model_idx = int(user_input) - 1
            if 0 <= model_idx < len(model_names):
                model_key = model_names[model_idx]
                print(f"\n模型 '{model_key}' 的详细信息:")
                
                # 打印评估指标
                if 'metrics' in data[model_key]:
                    print("评估指标:")
                    for metric_name, value in data[model_key]['metrics'].items():
                        print(f"  {metric_name}: {value}")
                
                # 打印预测值和真实值的统计信息
                pred_values = extract_predicted_values(data, model_key)
                true_values = extract_true_values(data, model_key)
                
                if pred_values is not None:
                    print(f"\n预测值统计 (形状 {pred_values.shape}):")
                    print(f"  均值: {np.mean(pred_values):.4f}")
                    print(f"  标准差: {np.std(pred_values):.4f}")
                    print(f"  最小值: {np.min(pred_values):.4f}")
                    print(f"  最大值: {np.max(pred_values):.4f}")
                
                if true_values is not None:
                    print(f"\n真实值统计 (形状 {true_values.shape}):")
                    print(f"  均值: {np.mean(true_values):.4f}")
                    print(f"  标准差: {np.std(true_values):.4f}")
                    print(f"  最小值: {np.min(true_values):.4f}")
                    print(f"  最大值: {np.max(true_values):.4f}")
            else:
                print("无效的模型编号，请重试")
        except ValueError:
            print("请输入有效的数字或 'q'")

if __name__ == "__main__":
    main() 