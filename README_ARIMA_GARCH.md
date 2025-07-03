# ARIMA+GARCH基准模型与TimesNet比较

本项目实现了基于ARIMA+GARCH系列模型的英镑兑人民币汇率预测基准模型，并提供了与TimesNet模型进行比较的工具。

## 模型列表

本项目实现了以下GARCH族模型作为基准：

1. **AR(p) + ARCH(1)**: 自回归+自回归条件异方差模型
2. **GARCH(1,1)**: 广义自回归条件异方差模型
3. **EGARCH(1,1)**: 指数广义自回归条件异方差模型
4. **GARCH-M(1,1)**: 均值中包含GARCH项的广义自回归条件异方差模型

## 目录结构

```
.
├── arima_garch_baseline.py      # ARIMA+GARCH基准模型实现
├── run_arima_garch.py           # 运行ARIMA+GARCH基准模型的脚本
├── compare_results.py           # 比较ARIMA+GARCH与TimesNet结果的脚本
├── run_comparison.sh            # 运行对比实验的Shell脚本
└── README_ARIMA_GARCH.md        # 本说明文件
```

## 安装依赖

在运行模型前，需要安装以下依赖：

```bash
pip install pandas numpy matplotlib seaborn statsmodels arch scikit-learn tqdm
```

## 使用方法

### 1. 准备数据

确保英镑兑人民币汇率数据 `英镑兑人民币_20250324_102930.csv` 位于 `./dataset/` 目录下。数据必须包含 `date` 和 `rate` 两列。

### 2. 运行ARIMA+GARCH基准模型

```bash
python run_arima_garch.py \
  --root_path ./dataset/ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --target rate \
  --seq_len 96 \
  --step_size 1
```

参数说明：
- `root_path`: 数据根目录
- `data_path`: 数据文件路径
- `target`: 目标变量列名
- `seq_len`: 用于训练的历史数据长度
- `step_size`: 滚动窗口的步长

结果将保存在 `arima_garch_results` 目录下。

### 3. 运行TimesNet模型

首先运行TimesNet模型进行英镑兑人民币汇率预测：

```bash
bash scripts/long_term_forecast/GBP_CNY_TimesNet.sh
```

TimesNet模型的结果将保存在 `./checkpoints/` 目录下。

### 4. 比较两种模型的结果

```bash
python compare_results.py \
  --arima_garch_dir arima_garch_results \
  --timesnet_dir ./checkpoints \
  --output_dir comparison_results
```

参数说明：
- `arima_garch_dir`: ARIMA+GARCH结果目录
- `timesnet_dir`: TimesNet检查点目录
- `output_dir`: 输出目录

比较结果将保存在 `comparison_results` 目录下。

### 5. 一键运行整个对比实验

或者，您可以使用提供的Shell脚本一键运行整个对比实验：

```bash
bash run_comparison.sh
```

## 模型详细信息

### ARIMA模型

ARIMA(p,d,q)模型是自回归整合移动平均模型，其中：
- p: 自回归项的阶数
- d: 差分阶数 (本项目固定为0)
- q: 移动平均项的阶数

### GARCH族模型

GARCH族模型用于建模时间序列的条件异方差（波动率）。本项目实现了以下变体：

#### ARCH(1)
最简单的异方差自回归条件模型，波动率仅依赖于过去的残差平方。

#### GARCH(1,1)
广义自回归条件异方差模型，波动率依赖于过去的残差平方和过去的波动率。

#### EGARCH(1,1)
指数GARCH模型，能够捕捉杠杆效应，即不同符号的冲击对波动率的不对称影响。

#### GARCH-M(1,1)
均值方程中包含GARCH项的模型，允许风险（波动率）直接影响均值。

## 结果解读

比较结果包括：

1. **CSV表格** (`model_comparison.csv`): 包含所有模型在不同预测长度下的MSE、RMSE、MAE、MAPE评估指标。
2. **图表** (`model_comparison.png`): 直观展示不同模型在各预测长度下的性能比较。

## 注意事项

1. GARCH族模型可能在某些情况下拟合失败，此时代码会回退到仅使用ARIMA模型进行预测。
2. 为保持与TimesNet一致，模型使用96天的历史数据训练，预测未来30、60、90和180天的汇率。
3. 代码会自动寻找最佳的ARIMA参数(p,q)组合。 