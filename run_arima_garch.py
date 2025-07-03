#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行ARIMA+GARCH基准模型
"""

import os
import subprocess
import argparse
import time
import sys

def main(args):
    """
    主函数，运行不同配置的ARIMA+GARCH模型
    """
    # 确保结果目录存在
    os.makedirs('arima_garch_results', exist_ok=True)
    
    # 开始时间
    start_time = time.time()
    
    # 定义不同的AR参数
    ar_params = [(1, 0), (2, 0), (1, 1)]
    
    print(f"{'='*50}")
    print(f"开始运行ARIMA+GARCH基准模型")
    print(f"{'='*50}")
    
    best_rmse = float('inf')
    best_params = None
    all_output = ""

    for p, q in ar_params:
        print(f"\n运行 ARIMA({p},0,{q}) + GARCH 模型")

        # 构建命令
        cmd = [
            sys.executable, "arima_garch_baseline.py",
            "--root_path", args.root_path,
            "--data_path", args.data_path,
            "--target", args.target,
            "--p", str(p),
            "--q", str(q),
            "--seq_len", str(args.seq_len),
            "--step_size", str(args.step_size)
        ]

        # 运行命令并实时打印输出
        # 使用utf-8编码，行缓冲，忽略解码错误
        # 修改 subprocess.Popen，让子进程的输出直接打印到控制台
        # 移除 stdout=subprocess.PIPE, stderr=subprocess.PIPE
        print(f"正在执行命令: {' '.join(cmd)}") # 打印将要执行的命令
        process = subprocess.Popen(cmd, text=True, encoding='utf-8', errors='ignore', env={**os.environ, "PYTHONIOENCODING": "UTF-8"})

        # 等待子进程完成
        process.wait()
        
        # 由于输出直接打印，我们无法捕获 current_run_output 和 error 来进行后续的 RMSE 解析
        # 这部分逻辑需要调整，或者在 arima_garch_baseline.py 中将结果写入文件/标准输出特定格式
        # 为了诊断卡顿问题，我们暂时牺牲这部分功能
        current_run_output = "" # 无法捕获，暂时设为空
        all_output += f"[INFO] Output for ARIMA({p},0,{q}) was streamed directly to console.\n" # 记录一下
        error = "" # 无法捕获，暂时设为空

        if process.returncode != 0:
            print(f"\n运行 ARIMA({p},0,{q}) 时子进程出错，返回码: {process.returncode}")
        else:
            print(f"\nARIMA({p},0,{q}) 子进程运行完成。")
        
        # 解析当前运行的输出，查找30天预测的RMSE结果
        # !!! 注意: 以下RMSE解析逻辑将无法正常工作，因为 current_run_output 为空 !!!
        # !!! 这部分代码保留结构，但预期不会找到RMSE            !!!
        lines = current_run_output.split('\n')
        found_pred_len_30_section = False
        rmse_for_30_days = None

        for line in lines:
            line = line.strip()
            if "预测长度: 30天" in line:
                found_pred_len_30_section = True
                continue

            if found_pred_len_30_section:
                if "预测长度:" in line or "模型:" in line or "----" in line or "====" in line:
                    found_pred_len_30_section = False
                    if rmse_for_30_days is not None:
                        break
                
                if 'RMSE:' in line and rmse_for_30_days is None:
                    try:
                        rmse_val = float(line.split(':')[1].strip())
                        rmse_for_30_days = rmse_val
                        print(f"找到 ARIMA({p},0,{q}) 的 30天预测 RMSE: {rmse_for_30_days:.4f}")
                        if rmse_for_30_days < best_rmse:
                            best_rmse = rmse_for_30_days
                            best_params = (p, q)
                            print(f"新的最佳参数: ARIMA({p},0,{q})，30天预测RMSE = {best_rmse:.4f}")
                        break
                    
                    except ValueError:
                        print(f"无法解析RMSE值从行: {line}")
                    except IndexError:
                        print(f"无法从行分割RMSE值: {line}")
                    except Exception as e:
                        print(f"解析RMSE时发生未知错误: {e}, 行: {line}")
                        pass
        
        if rmse_for_30_days is None:
            print(f"警告: 未能在 ARIMA({p},0,{q}) 的输出中找到有效的30天预测RMSE。")

    # 计算总运行时间
    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"所有模型运行完成，总耗时: {elapsed_time:.2f}秒")
    if best_params:
        print(f"最佳ARIMA参数: ({best_params[0]},0,{best_params[1]})，基于30天预测RMSE = {best_rmse:.4f}")
    else:
        print("未能确定最佳参数（在测试的配置中未找到有效的30天RMSE）。")
    print(f"{'='*50}")
    
    # 使用最佳参数再次运行所有预测长度
    if best_params:
        print("\n使用最佳参数再次运行所有预测长度...")
        p, q = best_params
        
        # 构建命令
        cmd = [
            sys.executable, "arima_garch_baseline.py",
            "--root_path", args.root_path,
            "--data_path", args.data_path,
            "--target", args.target,
            "--p", str(p),
            "--q", str(q),
            "--seq_len", str(args.seq_len),
            "--step_size", str(args.step_size)
        ]
        
        # 运行命令并实时打印输出
        # 使用utf-8编码，行缓冲，忽略解码错误
        # 修改 subprocess.Popen，让子进程的输出直接打印到控制台
        print(f"正在执行命令: {' '.join(cmd)}") # 打印将要执行的命令
        process = subprocess.Popen(cmd, text=True, encoding='utf-8', errors='ignore', env={**os.environ, "PYTHONIOENCODING": "UTF-8"})

        # 等待子进程完成
        process.wait()

        if process.returncode != 0:
            print(f"\n使用最佳参数 ({p},0,{q}) 运行时子进程出错，返回码: {process.returncode}")
        else:
            print("\n最佳参数模型运行完成 (输出已直接打印到控制台)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行ARIMA+GARCH基准模型')
    
    # 数据参数
    parser.add_argument('--root_path', type=str, default='./dataset/', help='数据根目录')
    parser.add_argument('--data_path', type=str, default='英镑兑人民币_20250324_102930.csv', help='数据文件路径')
    parser.add_argument('--target', type=str, default='rate', help='目标变量列名')
    
    # 模型参数
    parser.add_argument('--seq_len', type=int, default=96, help='用于训练的历史数据长度')
    parser.add_argument('--step_size', type=int, default=1, help='滚动窗口的步长')
    
    args = parser.parse_args()
    
    main(args) 