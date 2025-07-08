# ARIMA+GARCH 时间序列预测系统说明文档

## 📋 系统概述

这是一个专门用于汇率时间序列预测的 ARIMA+GARCH 模型系统，特别针对英镑兑人民币（GBPCNY）汇率预测进行了优化。系统支持多步预测、多种模型架构和灵活的参数配置。

### 🎯 主要功能
- **多步预测支持**：1天、24天、30天、60天、90天、180天预测
- **多种模型类型**：
  - Naive Baseline：朴素基线模型
  - Pure ARMA：纯时间序列模型
  - GARCH族模型：ARCH、GARCH、EGARCH、GARCH-M、GJR-GARCH、TGARCH
- **两种均值模型**：Constant均值和AR均值
- **动态测试集**：使用15%数据作为测试集
- **全面诊断**：Ljung-Box检验、正态性检验、残差分析

## 🛠️ 环境要求

### Python 版本
- Python 3.7 或更高版本

### 依赖包
```bash
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
statsmodels>=0.12.0
arch>=4.19.0
scipy>=1.6.0
tqdm>=4.60.0
seaborn>=0.11.0
```

### 安装依赖
```bash
pip install numpy pandas matplotlib scikit-learn statsmodels arch scipy tqdm seaborn
```

## 📁 文件结构

```
Time-Series-Library-main/
├── run_arima_garch_jpy_last150test.py  # 主程序文件
├── dataset/
│   └── 英镑兑人民币_short_series.csv   # 数据文件
└── README_ARIMA_GARCH.md              # 说明文档（本文件）
```

## 🚀 快速开始

### 基本用法
```bash
# 使用默认配置运行（包含所有预测步长和ARMA参数）
python run_arima_garch_jpy_last150test.py
```

### 常用配置
```bash
# 跳过1步预测，只保留p1q1参数组合
python run_arima_garch_jpy_last150test.py --pred_lens 24 30 60 90 180 --arma_p_values 1 --arma_q_values 1

# 只测试特定预测步长
python run_arima_garch_jpy_last150test.py --pred_lens 30 60 90

# 自定义训练窗口大小
python run_arima_garch_jpy_last150test.py --seq_lens 1000 500
```

## 📖 命令行参数详解

### 基础参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--root_path` | str | `./dataset/` | 数据根目录 |
| `--data_path` | str | `英镑兑人民币_short_series.csv` | 数据文件路径 |
| `--target` | str | `Close` | 目标变量列名 |
| `--step_size` | int | `1` | 滚动窗口步长 |

### 模型配置参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--pred_lens` | int+ | `[1,24,30,60,90,180]` | 预测步长列表 |
| `--arma_p_values` | int+ | `[1,1,2]` | ARMA模型p参数列表 |
| `--arma_q_values` | int+ | `[0,1,0]` | ARMA模型q参数列表 |
| `--seq_lens` | int+ | `[1000,500,250,125]` | 训练窗口大小列表 |

### 参数说明
- `pred_lens`：预测步长，支持1-180天的多步预测
- `arma_p_values` 和 `arma_q_values`：必须长度相同，配对使用形成ARMA(p,q)参数组合
- `seq_lens`：滚动窗口中使用的历史数据长度

## 📊 输出结果

### 目录结构
系统会为每个配置组合创建独立的结果目录：
```
arima_garch_results_logret_GBPCNY_seq{窗口长度}_{预测步长}step_last15pct_test/
├── model_params_GBPCNY_seq{窗口长度}_{预测步长}step.json
├── results_GBPCNY_seq{窗口长度}_pred_len_{预测步长}.pkl
└── summary_table_seq{窗口长度}_{预测步长}step.csv
```

### 文件说明

#### 1. JSON文件（模型参数）
保存第一个窗口的模型参数和诊断信息：
```json
{
  "Pure ARMA(1,1)": {
    "model_type": "Pure ARMA(1,1)",
    "parameters": {
      "mean_equation": {
        "ar": [0.123],
        "const": 0.456
      }
    }
  }
}
```

#### 2. PKL文件（完整数据）
包含所有模型的详细结果：
- 评估指标（MSE, RMSE, MAE, MAX_AE, MAPE, R²）
- 真实值和预测值数组
- 执行时间
- Ljung-Box检验结果

#### 3. CSV文件（汇总表格）
所有模型在该配置下的性能对比表格

### 评估指标
| 指标 | 说明 |
|------|------|
| MSE | 均方误差 |
| RMSE | 均方根误差 |
| MAE | 平均绝对误差 |
| MAX_AE | 最大绝对误差 |
| MAPE | 平均绝对百分比误差 |
| R² | 决定系数 |

## 💡 使用示例

### 示例1：快速测试（小规模）
```bash
# 只测试30天预测，使用p1q1，一个窗口大小
python run_arima_garch_jpy_last150test.py \
  --pred_lens 30 \
  --arma_p_values 1 \
  --arma_q_values 1 \
  --seq_lens 500
```

### 示例2：跳过短期预测
```bash
# 只测试中长期预测（60天以上）
python run_arima_garch_jpy_last150test.py \
  --pred_lens 60 90 180
```

### 示例3：比较不同ARMA参数
```bash
# 比较三种不同的ARMA参数组合
python run_arima_garch_jpy_last150test.py \
  --pred_lens 30 \
  --arma_p_values 1 1 2 \
  --arma_q_values 0 1 0
```

### 示例4：自定义数据文件
```bash
# 使用自定义数据文件和目标列
python run_arima_garch_jpy_last150test.py \
  --data_path my_currency_data.csv \
  --target price \
  --pred_lens 24 48
```

## ⚙️ 高级配置

### 模型选择策略
- **1步预测**：使用全部GARCH架构（包括EGARCH和TGARCH）
- **多步预测**：自动排除不稳定架构（EGARCH、TGARCH）

### 数据预处理
- 自动计算对数收益率
- Z-score标准化
- 平稳性检验（ADF测试）

### 异常处理
- 极端值检测和处理（阈值：10.0）
- 自动回退到Pure ARMA模型
- NaN/Inf值清理

## 🔧 故障排除

### 常见错误

#### 1. 数据加载错误
```
错误: 目标列 'Close' 在数据中未找到
```
**解决方案**：检查数据文件中的列名，使用`--target`参数指定正确的列名。

#### 2. 参数长度不匹配
```
错误: arma_p_values和arma_q_values长度必须相同
```
**解决方案**：确保p值和q值列表长度相同：
```bash
--arma_p_values 1 2 --arma_q_values 0 1  # 正确
--arma_p_values 1 2 3 --arma_q_values 0 1  # 错误
```

#### 3. 内存不足
对于大数据集或长训练窗口，可能出现内存不足。
**解决方案**：
- 减少训练窗口大小：`--seq_lens 250 125`
- 减少预测步长：`--pred_lens 30 60`

### 性能优化建议

1. **减少配置组合**：
   ```bash
   # 较少的配置组合
   --pred_lens 30 60 --arma_p_values 1 --arma_q_values 1
   ```

2. **使用较小的训练窗口**：
   ```bash
   --seq_lens 500 250
   ```

3. **分批运行**：
   ```bash
   # 先运行短期预测
   --pred_lens 24 30
   # 再运行长期预测  
   --pred_lens 90 180
   ```

## 📈 结果分析

### 查看运行状态
程序运行时会显示：
- 配置信息
- 数据加载状态
- 平稳性检验结果
- 每个模型的进度条
- 实时评估指标

### 结果比较
程序结束时会自动生成：
- 按预测步长分组的结果排名
- 全局最佳模型配置
- 详细的性能指标对比

## 📝 注意事项

1. **数据格式要求**：
   - CSV格式
   - 必须包含日期列（date或Date）
   - 目标列为数值型且大于0

2. **计算时间**：
   - 完整配置可能需要数小时
   - 建议先用小规模配置测试

3. **磁盘空间**：
   - 每个配置会生成3个文件
   - 确保有足够的存储空间

4. **模型稳定性**：
   - EGARCH和TGARCH在多步预测中可能不稳定
   - 系统会自动处理并回退到稳定模型

## 🤝 技术支持

如果遇到问题，请检查：
1. Python版本和依赖包是否正确安装
2. 数据文件路径和格式是否正确
3. 命令行参数是否符合要求
4. 是否有足够的内存和磁盘空间

## 🧠 深度学习模型快速使用说明

本系统还支持多种主流深度学习时序预测模型，包括 Autoformer、iTransformer、TimesNet、TimeMixer 等，适合复杂或长期预测任务。

### 支持模型
- Autoformer
- iTransformer
- TimesNet
- TimeMixer

### 数据要求
- CSV格式，含日期列（date/Date）和目标列（如 rate）
- 放在 `./dataset/` 目录下，例如：`英镑兑人民币_20250324_102930.csv`

### 一键运行方法
进入项目根目录，执行如下命令（以Autoformer为例）：

```bash
bash scripts/long_term_forecast/GBP_CNY_Autoformer.sh
```

如需运行其他模型，只需替换脚本名：
- iTransformer: `bash scripts/long_term_forecast/GBP_CNY_iTransformer.sh`
- TimesNet: `bash scripts/long_term_forecast/GBP_CNY_TimesNet.sh`
- TimeMixer: `bash scripts/long_term_forecast/GBP_CNY_TimeMixer.sh`

脚本会自动完成30天、60天、90天、180天等多步预测。

### 手动运行示例（以Autoformer为例）

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --model_id GBP_CNY_96_30 \
  --model Autoformer \
  --data gbp_cny \
  --features S \
  --target rate \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --freq d \
  --inverse
```

不同模型参数略有差异，详见`scripts/long_term_forecast/`下各脚本。

### 结果查看
- 结果保存在 `results/` 目录下，每个模型和预测步长有独立子目录
- 主要评估指标保存在 `metrics.npy` 和自动生成的 `metrics.csv`
- 可用如下命令批量汇总所有模型结果：

```bash
python convert_metrics_to_csv.py
```

将在 `results/all_metrics_summary.csv` 生成总览表，便于模型对比。

---
如需自定义参数或了解更多细节，请参考各脚本和 `run.py` 注释。

---
