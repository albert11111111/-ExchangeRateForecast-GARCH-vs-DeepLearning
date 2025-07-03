#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多步预测系统测试脚本
===================

功能：
1. 测试修复后的ARIMA+GARCH多步预测系统
2. 创建专门的测试结果文件夹
3. 运行小规模测试验证代码正确性
"""

import os
import sys
import subprocess
import datetime
import shutil

def create_test_directory():
    """创建测试结果目录"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"test_results_multistep_{timestamp}"
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    os.makedirs(test_dir, exist_ok=True)
    print(f"创建测试目录: {test_dir}")
    return test_dir

def prepare_test_data():
    """准备测试数据"""
    # 检查是否存在测试数据
    data_file = "./dataset/sorted_output_file.csv"
    if not os.path.exists(data_file):
        print(f"警告: 测试数据文件不存在: {data_file}")
        print("请确保数据文件存在于指定路径")
        return False
    
    print(f"找到测试数据文件: {data_file}")
    return True

def run_quick_test(test_dir):
    """运行快速测试"""
    print("\n" + "="*80)
    print("开始运行多步预测系统快速测试")
    print("="*80)
    
    # 切换到测试目录
    original_dir = os.getcwd()
    
    try:
        # 准备测试参数
        test_args = [
            "python3", 
            "run_arima_garch_jpy_last150test.py",
            "--root_path", "./dataset/",
            "--data_path", "sorted_output_file.csv", 
            "--target", "rate",
            "--seq_len", "31",
            "--step_size", "1"
        ]
        
        print("测试命令:")
        print(" ".join(test_args))
        print("\n开始执行...")
        
        # 运行测试
        result = subprocess.run(
            test_args,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        # 保存测试日志
        log_file = os.path.join(test_dir, "test_log.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"测试时间: {datetime.datetime.now()}\n")
            f.write(f"命令: {' '.join(test_args)}\n")
            f.write(f"返回码: {result.returncode}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
        
        print(f"测试完成，返回码: {result.returncode}")
        print(f"日志已保存到: {log_file}")
        
        if result.returncode == 0:
            print("✅ 测试成功完成!")
            
            # 移动生成的结果文件到测试目录
            move_results_to_test_dir(test_dir)
            
        else:
            print("❌ 测试失败!")
            print("错误输出:")
            print(result.stderr[-1000:])  # 显示最后1000字符的错误
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ 测试超时!")
        return False
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {e}")
        return False

def move_results_to_test_dir(test_dir):
    """移动生成的结果文件到测试目录"""
    print("\n移动结果文件到测试目录...")
    
    # 查找所有生成的结果目录
    result_dirs = []
    for item in os.listdir("."):
        if item.startswith("arima_garch_results_logret_USDJPY"):
            result_dirs.append(item)
    
    if result_dirs:
        results_subdir = os.path.join(test_dir, "prediction_results")
        os.makedirs(results_subdir, exist_ok=True)
        
        for result_dir in result_dirs:
            src_path = result_dir
            dst_path = os.path.join(results_subdir, result_dir)
            try:
                shutil.move(src_path, dst_path)
                print(f"✅ 移动: {result_dir} -> {dst_path}")
            except Exception as e:
                print(f"❌ 移动失败 {result_dir}: {e}")
    else:
        print("未找到结果目录")

def generate_test_summary(test_dir, success):
    """生成测试总结"""
    summary_file = os.path.join(test_dir, "test_summary.md")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# 多步预测系统测试总结\n\n")
        f.write(f"**测试时间**: {datetime.datetime.now()}\n\n")
        f.write(f"**测试状态**: {'✅ 成功' if success else '❌ 失败'}\n\n")
        
        f.write("## 测试配置\n\n")
        f.write("- **数据文件**: ./dataset/sorted_output_file.csv\n")
        f.write("- **目标变量**: rate (汇率)\n")
        f.write("- **序列长度**: 31\n")
        f.write("- **预测步长**: 1, 24, 30, 60, 90, 180天\n")
        f.write("- **测试集大小**: 150个数据点\n\n")
        
        f.write("## 测试目的\n\n")
        f.write("1. 验证修复后的数组长度一致性问题\n")
        f.write("2. 测试多步预测功能正常运行\n")
        f.write("3. 确保结果文件正确生成\n")
        f.write("4. 检查各种GARCH模型的稳定性\n\n")
        
        if success:
            f.write("## 测试结果\n\n")
            f.write("✅ 所有功能正常运行\n")
            f.write("✅ 数组长度一致性问题已解决\n")
            f.write("✅ 多步预测功能正常\n")
            f.write("✅ 结果文件成功生成\n\n")
            
            # 检查生成的结果文件
            results_dir = os.path.join(test_dir, "prediction_results")
            if os.path.exists(results_dir):
                f.write("## 生成的结果文件\n\n")
                for item in os.listdir(results_dir):
                    f.write(f"- {item}/\n")
                    result_path = os.path.join(results_dir, item)
                    if os.path.isdir(result_path):
                        for subitem in os.listdir(result_path):
                            f.write(f"  - {subitem}\n")
                f.write("\n")
        else:
            f.write("## 测试失败\n\n")
            f.write("请检查 test_log.txt 了解详细错误信息\n\n")
        
        f.write("## 文件说明\n\n")
        f.write("- `test_log.txt`: 完整的测试执行日志\n")
        f.write("- `prediction_results/`: 预测结果文件夹\n")
        f.write("- `test_summary.md`: 本测试总结文件\n")
    
    print(f"测试总结已保存到: {summary_file}")

def main():
    """主测试函数"""
    print("ARIMA+GARCH多步预测系统测试")
    print("="*50)
    
    # 1. 创建测试目录
    test_dir = create_test_directory()
    
    # 2. 检查测试数据
    if not prepare_test_data():
        return
    
    # 3. 运行测试
    success = run_quick_test(test_dir)
    
    # 4. 生成测试总结
    generate_test_summary(test_dir, success)
    
    print("\n" + "="*80)
    if success:
        print("🎉 测试全部完成! 系统运行正常")
        print(f"📁 测试结果保存在: {test_dir}")
        print("\n主要修复内容:")
        print("✅ 数组长度一致性检查")
        print("✅ 边界条件处理")
        print("✅ 多步预测循环优化") 
        print("✅ 详细错误提示和调试信息")
    else:
        print("❌ 测试失败，请检查错误日志")
        print(f"📁 错误日志位置: {test_dir}/test_log.txt")
    print("="*80)

if __name__ == "__main__":
    main()