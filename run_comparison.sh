#!/bin/bash

# 设置工作目录
cd new-Timeserires/Time-Series-Library-main/Time-Series-Library-main/

# 创建结果目录
mkdir -p arima_garch_results
mkdir -p comparison_results

echo "========================================"
echo "开始英镑兑人民币汇率预测对比实验"
echo "========================================"

# 1. 运行ARIMA+GARCH模型
echo "1. 运行ARIMA+GARCH基准模型..."
python run_arima_garch.py \
  --root_path ./dataset/ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --target rate \
  --seq_len 96 \
  --step_size 1

if [ $? -ne 0 ]; then
  echo "ARIMA+GARCH模型运行失败！"
  exit 1
fi

# 2. 运行模型比较
echo "2. 比较TimesNet和ARIMA+GARCH模型结果..."
python compare_results.py \
  --arima_garch_dir arima_garch_results \
  --timesnet_dir ./checkpoints \
  --output_dir comparison_results

if [ $? -ne 0 ]; then
  echo "模型比较失败！"
  exit 1
fi

echo "========================================"
echo "实验完成！结果保存在comparison_results目录"
echo "========================================" 