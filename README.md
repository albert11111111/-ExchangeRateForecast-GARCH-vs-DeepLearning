# Time Series Library (TSLib)
TSLib is an open-source library for deep learning researchers, especially for deep time series analysis.

We provide a neat code base to evaluate advanced deep time series models or develop your model, which covers five mainstream tasks: **long- and short-term forecasting, imputation, anomaly detection, and classification.**

:triangular_flag_on_post:**News** (2024.10) We have included [[TimeXer]](https://arxiv.org/abs/2402.19072), which defined a practical forecasting paradigm: Forecasting with Exogenous Variables. Considering both practicability and computation efficiency, we believe the new forecasting paradigm defined in TimeXer can be the "right" task for future research.

:triangular_flag_on_post:**News** (2024.10) Our lab has open-sourced [[OpenLTM]](https://github.com/thuml/OpenLTM), which provides a distinct pretrain-finetuning paradigm compared to TSLib. If you are interested in Large Time Series Models, you may find this repository helpful.

:triangular_flag_on_post:**News** (2024.07) We wrote a comprehensive survey of [[Deep Time Series Models]](https://arxiv.org/abs/2407.13278) with a rigorous benchmark based on TSLib. In this paper, we summarized the design principles of current time series models supported by insightful experiments, hoping to be helpful to future research.

:triangular_flag_on_post:**News** (2024.04) Many thanks for the great work from [frecklebars](https://github.com/thuml/Time-Series-Library/pull/378). The famous sequential model [Mamba](https://arxiv.org/abs/2312.00752) has been included in our library. See [this file](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py), where you need to install `mamba_ssm` with pip at first.

:triangular_flag_on_post:**News** (2024.03) Given the inconsistent look-back length of various papers, we split the long-term forecasting in the leaderboard into two categories: Look-Back-96 and Look-Back-Searching. We recommend researchers read [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2), which includes both look-back length settings in experiments for scientific rigor.

:triangular_flag_on_post:**News** (2023.10) We add an implementation to [iTransformer](https://arxiv.org/abs/2310.06625), which is the state-of-the-art model for long-term forecasting. The official code and complete scripts of iTransformer can be found [here](https://github.com/thuml/iTransformer).

:triangular_flag_on_post:**News** (2023.09) We added a detailed [tutorial](https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb) for [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq) and this library, which is quite friendly to beginners of deep time series analysis.

:triangular_flag_on_post:**News** (2023.02) We release the TSlib as a comprehensive benchmark and code base for time series models, which is extended from our previous GitHub repository [Autoformer](https://github.com/thuml/Autoformer).

## Leaderboard for Time Series Analysis

Till March 2024, the top three models for five different tasks are:

| Model<br>Ranking | Long-term<br>Forecasting<br>Look-Back-96              | Long-term<br/>Forecasting<br/>Look-Back-Searching     | Short-term<br>Forecasting                                    | Imputation                                                   | Classification                                               | Anomaly<br>Detection                               |
| ---------------- | ----------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| 🥇 1st            | [TimeXer](https://arxiv.org/abs/2402.19072)      | [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2) | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)       |
| 🥈 2nd            | [iTransformer](https://arxiv.org/abs/2310.06625) | [PatchTST](https://github.com/yuqinie98/PatchTST)     | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [FEDformer](https://github.com/MAZiqing/FEDformer) |
| 🥉 3rd            | [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2)          | [DLinear](https://arxiv.org/pdf/2205.13504.pdf)       | [FEDformer](https://github.com/MAZiqing/FEDformer)           | [Autoformer](https://github.com/thuml/Autoformer)            | [Informer](https://github.com/zhouhaoyi/Informer2020)        | [Autoformer](https://github.com/thuml/Autoformer)  |


**Note: We will keep updating this leaderboard.** If you have proposed advanced and awesome models, you can send us your paper/code link or raise a pull request. We will add them to this repo and update the leaderboard as soon as possible.

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.
  - [x] **TimeXer** - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [[NeurIPS 2024]](https://arxiv.org/abs/2402.19072) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py)
  - [x] **TimeMixer** - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [[ICLR 2024]](https://openreview.net/pdf?id=7oLshfEIC2) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py).
  - [x] **TSMixer** - TSMixer: An All-MLP Architecture for Time Series Forecasting [[arXiv 2023]](https://arxiv.org/pdf/2303.06053.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TSMixer.py)
  - [x] **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py).
  - [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py).
  - [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py).
  - [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py).
  - [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py).
  - [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py).
  - [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py).
  - [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py).
  - [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py).
  - [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py).
  - [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py).

See our latest paper [[TimesNet]](https://arxiv.org/abs/2210.02186) for the comprehensive benchmark. We will release a real-time updated online version soon.

**Newly added baselines.** We will add them to the leaderboard after a comprehensive evaluation.
  - [x] **MultiPatchFormer** - A multiscale model for multivariate time series forecasting [[Scientific Reports 2025]](https://www.nature.com/articles/s41598-024-82417-4) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MultiPatchFormer.py)
  - [x] **WPMixer** - WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting [[AAAI 2025]](https://arxiv.org/abs/2412.17176) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/WPMixer.py)
  - [x] **PAttn** - Are Language Models Actually Useful for Time Series Forecasting? [[NeurIPS 2024]](https://arxiv.org/pdf/2406.16964) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PAttn.py)
  - [x] **Mamba** - Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[arXiv 2023]](https://arxiv.org/abs/2312.00752) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py)
  - [x] **SegRNN** - SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting [[arXiv 2023]](https://arxiv.org/abs/2308.11200.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/SegRNN.py).
  - [x] **Koopa** - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors [[NeurIPS 2023]](https://arxiv.org/pdf/2305.18803.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Koopa.py).
  - [x] **FreTS** - Frequency-domain MLPs are More Effective Learners in Time Series Forecasting [[NeurIPS 2023]](https://arxiv.org/pdf/2311.06184.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FreTS.py).
  - [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py).
  - [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py).
  - [x] **TiDE** - Long-term Forecasting with TiDE: Time-series Dense Encoder [[arXiv 2023]](https://arxiv.org/pdf/2304.08424.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TiDE.py).
  - [x] **SCINet** - SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction [[NeurIPS 2022]](https://openreview.net/pdf?id=AyajSjTAzmg)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/SCINet.py).
  - [x] **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/forum?id=zTQdHSQUQWc)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py).
  - [x] **TFT** - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting [[arXiv 2019]](https://arxiv.org/abs/1912.09363)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TemporalFusionTransformer.py). 
 
## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. Here is a summary of supported datasets.

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# classification
bash ./scripts/classification/TimesNet.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

Note: 

(1) About classification: Since we include all five tasks in a unified code base, the accuracy of each subtask may fluctuate but the average performance can be reproduced (even a bit better). We have provided the reproduced checkpoints [here](https://github.com/thuml/Time-Series-Library/issues/494).

(2) About anomaly detection: Some discussion about the adjustment strategy in anomaly detection can be found [here](https://github.com/thuml/Anomaly-Transformer/issues/14). The key point is that the adjustment strategy corresponds to an event-level metric.

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}

@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```

## Contact
If you have any questions or suggestions, feel free to contact our maintenance team:

Current:
- Haixu Wu (Ph.D. student, wuhx23@mails.tsinghua.edu.cn)
- Yong Liu (Ph.D. student, liuyong21@mails.tsinghua.edu.cn)
- Yuxuan Wang (Ph.D. student, wangyuxu22@mails.tsinghua.edu.cn)
- Huikun Weng (Undergraduate, wenghk22@mails.tsinghua.edu.cn)

Previous:
- Tengge Hu (Master student, htg21@mails.tsinghua.edu.cn)
- Haoran Zhang (Master student, z-hr20@mails.tsinghua.edu.cn)
- Jiawei Guo (Undergraduate, guo-jw21@mails.tsinghua.edu.cn)

Or describe it in Issues.

## Acknowledgement

This project is supported by the National Key R&D Program of China (2021YFB1715200).

This library is constructed based on the following repos:

- Forecasting: https://github.com/thuml/Autoformer.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://github.com/thuml/Flowformer.

All the experiment datasets are public, and we obtain them from the following links:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer.

- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://www.timeseriesclassification.com/.

## All Thanks To Our Contributors

<a href="https://github.com/thuml/Time-Series-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=thuml/Time-Series-Library" />
</a>

# ARIMA+GARCH 时间序列预测模型优化版

本项目是对ARIMA+GARCH时间序列预测模型的优化实现，专注于汇率预测，支持单步和多步预测。

## 主要特性

- **完整模型族支持**：
  - 支持所有与ARMA结合的模型组合 (ARCH、GARCH、EGARCH、GARCH-M)
  - 支持AR和Constant两种均值设置
  - 支持多种ARMA参数组合 (p=1,2; q=0,1,2)
- **多步预测**：支持不同长度(1天、7天、14天、30天、90天、180天)的预测
- **一致的数据划分**：使用与baseline相同的70/15/15比例划分训练、验证、测试集
- **完善的异常处理**：增强了对极端值、NaN和Inf的处理
- **详细的评估指标**：MSE、RMSE、MAE、MAPE和R²
- **可视化结果**：自动生成预测结果图表和评估指标汇总表
- **CSV格式结果**：所有预测结果以CSV格式保存，便于后续分析和使用

## 主要改进

与原始baseline实现相比，优化版本有以下重要改进：

1. **全面的模型覆盖**：支持所有模型组合，包括纯ARMA、ARCH、GARCH、EGARCH、GARCH-M
2. **Constant均值支持**：新增了对Constant均值的支持，而不仅限于AR均值
3. **强化了GARCH-M模型**：特别对GARCH-M(1,1)_Const模型进行了改进，修复了之前的错误
4. **多步预测优化**：正确处理了多步预测时的误差累积效应
5. **长期预测能力**：支持多达180天的长期预测
6. **增强的稳健性**：改进了模型对异常值的处理和错误恢复机制
7. **参数优化**：更合理的默认参数设置，提高了模型的稳定性
8. **结果评估一致性**：与baseline完全一致的70/15/15数据划分方式
9. **结果存储优化**：以CSV格式保存预测结果，便于后续分析和使用

## 模型类型

支持的模型类型包括：

1. **AR均值模型组**：
   - ARCH(1)_AR
   - GARCH(1,1)_AR
   - EGARCH(1,1)_AR
   - GARCH-M(1,1)_AR

2. **Constant均值模型组**：
   - ARCH(1)_Const
   - GARCH(1,1)_Const
   - EGARCH(1,1)_Const
   - GARCH-M(1,1)_Const

3. **纯ARMA模型**：
   - Pure_ARMA_p_q (其中p和q取不同值)

4. **Naive Baseline**：
   - 使用前一天的值作为所有未来步的预测

## 使用方法

### 基本用法

```bash
python optimized_arima_garch_multistep.py --root_path ./dataset/ --data_path sorted_output_file.csv --target rate --seq_len 96
```

### 参数说明

- `--root_path`: 数据文件所在的根目录
- `--data_path`: 数据文件名
- `--target`: 目标变量列名（如汇率）
- `--seq_len`: 序列长度，用于训练模型的历史长度
- `--step_size`: 滚动窗口的步长，默认为1

## 结果文件

所有结果将保存在以下格式的目录中：
- Constant均值模型组: `arima_garch_results_logret_USDJPY_constant_seqXXX_multistep_ratio_70_15_15/`
- AR均值模型组(依赖ARMA参数): `arima_garch_results_logret_USDJPY_pX_qY_seqZZZ_multistep_ratio_70_15_15/`

其中每个目录包含：
- 预测结果CSV: `pred_prices_模型名称_lenN.csv`
- 真实值CSV: `true_prices_模型名称_lenN.csv`
- 评估指标汇总表: `summary_table_*.csv`
- 可视化图表: `plot_price_level_*.png`
