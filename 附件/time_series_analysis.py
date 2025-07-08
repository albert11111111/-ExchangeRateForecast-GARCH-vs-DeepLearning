#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat
from scipy import stats
import arch.unitroot as unitroot
from arch.unitroot import ADF
from statsmodels.stats.diagnostic import het_arch
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data():
    """加载数据"""
    root_path = './dataset/'
    data_path = '英镑兑人民币_20250324_102930.csv'
    df = pd.read_csv(f"{root_path}{data_path}")
    df['log_return_usd_jpy'] = np.log(df['rate']).diff()
    return df

def plot_time_series(df):
    """绘制原始时间序列图"""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['rate'], color='blue', alpha=0.7)
    plt.title('USD/JPY 汇率时间序列图')
    plt.xlabel('时间')
    plt.ylabel('汇率')
    plt.grid(True)
    plt.savefig('plots/1_原始时间序列.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_return_series(df):
    """绘制收益率序列图"""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['log_return_usd_jpy'], color='green', alpha=0.7)
    plt.title('GBP/RMB 对数收益率序列图')
    plt.xlabel('时间')
    plt.ylabel('对数收益率')
    plt.grid(True)
    plt.savefig('plots/2_对数收益率序列.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_distribution(df):
    """绘制收益率分布图与QQ图"""
    # 直方图
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df['log_return_usd_jpy'].dropna(), bins=50, stat='density', alpha=0.6)
    sns.kdeplot(data=df['log_return_usd_jpy'].dropna(), color='red')
    
    # 添加正态分布曲线作为参考
    x = np.linspace(df['log_return_usd_jpy'].min(), df['log_return_usd_jpy'].max(), 100)
    plt.plot(x, stats.norm.pdf(x, df['log_return_usd_jpy'].mean(), df['log_return_usd_jpy'].std()), 
             'r--', label='正态分布参考线')
    plt.title('收益率分布图')
    plt.xlabel('对数收益率')
    plt.ylabel('密度')
    plt.legend()
    plt.savefig('plots/3_收益率分布图.png', dpi=300, bbox_inches='tight')
    plt.close()

    # QQ图
    plt.figure(figsize=(12, 6))
    stats.probplot(df['log_return_usd_jpy'].dropna(), dist="norm", plot=plt)
    plt.title('收益率Q-Q图')
    plt.savefig('plots/4_QQ图.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_acf_pacf(df):
    """绘制ACF和PACF图"""
    # ACF图
    plt.figure(figsize=(12, 6))
    plot_acf(df['log_return_usd_jpy'].dropna(), lags=40)
    plt.title('收益率序列的自相关函数(ACF)图')
    plt.savefig('plots/5_ACF图.png', dpi=300, bbox_inches='tight')
    plt.close()

    # PACF图
    plt.figure(figsize=(12, 6))
    plot_pacf(df['log_return_usd_jpy'].dropna(), lags=40)
    plt.title('收益率序列的偏自相关函数(PACF)图')
    plt.savefig('plots/6_PACF图.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_volatility_clustering(df):
    """绘制波动率聚集效应图"""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['log_return_usd_jpy'].abs(), color='purple', alpha=0.7)
    plt.title('收益率绝对值序列图（波动率聚集效应）')
    plt.xlabel('时间')
    plt.ylabel('|收益率|')
    plt.grid(True)
    plt.savefig('plots/7_波动率聚集效应.png', dpi=300, bbox_inches='tight')
    plt.close()

def check_arch_effect(returns):
    """进行ARCH效应检验"""
    # 计算自相关系数
    acf_abs = acf(returns.abs().dropna(), nlags=10)
    acf_square = acf(returns.dropna()**2, nlags=10)
    
    print("\nARCH效应检验结果:")
    print("绝对收益率的自相关系数:")
    for i, value in enumerate(acf_abs):
        print(f"lag {i}: {value:.4f}")
    
    print("\n平方收益率的自相关系数:")
    for i, value in enumerate(acf_square):
        print(f"lag {i}: {value:.4f}")

def perform_stationarity_tests(returns):
    """进行平稳性检验"""
    # ADF检验
    adf_test = ADF(returns.dropna())
    print("\n增广迪基-福勒(ADF)检验结果:")
    print(adf_test.summary().as_text())

def plot_rolling_statistics(df):
    """绘制滚动统计图"""
    window = 30
    rolling_mean = df['log_return_usd_jpy'].rolling(window=window).mean()
    rolling_std = df['log_return_usd_jpy'].rolling(window=window).std()
    
    # 滚动均值
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, rolling_mean, color='blue', label=f'{window}日滚动均值')
    plt.title(f'{window}日滚动均值图')
    plt.xlabel('时间')
    plt.ylabel('滚动均值')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/8_滚动均值.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 滚动标准差
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, rolling_std, color='red', label=f'{window}日滚动标准差')
    plt.title(f'{window}日滚动标准差图')
    plt.xlabel('时间')
    plt.ylabel('滚动标准差')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/9_滚动标准差.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_normality_tests(returns):
    """进行正态性检验"""
    returns_clean = returns.dropna()
    
    # Jarque-Bera检验
    jb_stat, jb_pvalue = jarque_bera(returns_clean)
    
    # 计算偏度和峰度
    sk = skew(returns_clean)
    ku = kurtosis(returns_clean)
    
    print("\n正态性检验结果:")
    print(f"Jarque-Bera检验统计量: {jb_stat:.4f}")
    print(f"Jarque-Bera检验p值: {jb_pvalue:.4f}")
    print(f"偏度: {sk:.4f}")
    print(f"超额峰度: {ku:.4f}")
    
    if jb_pvalue < 0.05:
        print("结论：在5%的显著性水平下拒绝正态分布假设")
    else:
        print("结论：不能拒绝正态分布假设")

def perform_arch_lm_test(returns, lags=12):
    """进行ARCH LM检验"""
    returns_clean = returns.dropna()
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns_clean, nlags=lags)
    
    print(f"\nARCH LM检验结果 (滞后阶数={lags}):")
    print(f"LM统计量: {lm_stat:.4f}")
    print(f"LM检验p值: {lm_pvalue:.4f}")
    print(f"F统计量: {f_stat:.4f}")
    print(f"F检验p值: {f_pvalue:.4f}")
    
    if lm_pvalue < 0.05:
        print("结论：在5%的显著性水平下存在ARCH效应")
    else:
        print("结论：不存在显著的ARCH效应")

def check_leverage_effect(returns):
    """检验杠杆效应"""
    returns_clean = returns.dropna()
    # 计算滞后一期收益率与当前波动率的相关系数
    lagged_returns = returns_clean[:-1]
    current_volatility = np.abs(returns_clean[1:])
    
    correlation = np.corrcoef(lagged_returns, current_volatility)[0,1]
    
    print("\n杠杆效应检验:")
    print(f"滞后收益率与当前波动率的相关系数: {correlation:.4f}")
    if correlation < 0:
        print("存在负相关关系，表明可能存在杠杆效应")
    else:
        print("不存在明显的杠杆效应")

def check_serial_correlation(returns):
    """检验序列相关性"""
    returns_clean = returns.dropna()
    
    # Durbin-Watson检验
    # 首先进行简单的线性回归
    X = np.ones(len(returns_clean))
    model = OLS(returns_clean, X).fit()
    dw_stat = durbin_watson(model.resid)
    
    print("\n序列相关性检验:")
    print(f"Durbin-Watson统计量: {dw_stat:.4f}")
    
    if dw_stat < 1.5:
        print("存在正序列相关")
    elif dw_stat > 2.5:
        print("存在负序列相关")
    else:
        print("不存在显著的序列相关")

def calculate_descriptive_stats(returns):
    """计算描述性统计量"""
    returns_clean = returns.dropna()
    
    stats_dict = {
        '样本量': len(returns_clean),
        '均值': returns_clean.mean(),
        '标准差': returns_clean.std(),
        '最小值': returns_clean.min(),
        '最大值': returns_clean.max(),
        '中位数': returns_clean.median(),
        '偏度': skew(returns_clean),
        '峰度': kurtosis(returns_clean) + 3  # 注意：scipy的kurtosis是超额峰度
    }
    
    print("\n描述性统计:")
    for key, value in stats_dict.items():
        print(f"{key}: {value:.6f}")

def calculate_basic_stats(df):
    """计算汇率对数序列和对数差分序列的基本统计量"""
    # 计算对数序列
    log_series = np.log(df['rate'])
    
    # 计算对数差分序列
    log_diff = log_series.diff().dropna()
    
    # 计算Ljung-Box统计量
    def calculate_lb(series, lags=8):
        acf_values = acf(series, nlags=lags, fft=False)[1:]  # 去掉lag 0
        lb_stat, p_value = q_stat(acf_values, len(series))
        return lb_stat[-1], p_value[-1]  # 返回最后一个lag的统计量和p值
    
    # 计算基本统计量
    basic_stats = {
        '均值': [log_series.mean(), log_diff.mean()],
        '方差': [log_series.var(), log_diff.var()],
        '偏度': [skew(log_series), skew(log_diff)],
        '峰度': [kurtosis(log_series) + 3, kurtosis(log_diff) + 3],  # 加3是因为scipy返回的是超额峰度
        'L-B(8)': [
            calculate_lb(log_series)[0],
            calculate_lb(log_diff)[0]
        ],
        'ADF': [
            ADF(log_series).stat,
            ADF(log_diff).stat
        ],
        'LM（12）': [
            het_arch(log_series, nlags=12)[0],
            het_arch(log_diff, nlags=12)[0]
        ]
    }
    
    # 创建DataFrame
    df_stats = pd.DataFrame(basic_stats, index=['对数序列', '对数差分序列'])
    
    # 添加p值
    p_values = {
        'L-B(8)': [
            f"({format(calculate_lb(log_series)[1], '.4f')})",
            f"({format(calculate_lb(log_diff)[1], '.4f')})"
        ],
        'ADF': [
            f"({format(ADF(log_series).pvalue, '.4f')})",
            f"({format(ADF(log_diff).pvalue, '.4f')})"
        ],
        'LM（12）': [
            f"({format(het_arch(log_series, nlags=12)[1], '.4f')})",
            f"({format(het_arch(log_diff, nlags=12)[1], '.4f')})"
        ]
    }
    
    # 格式化输出
    formatted_df = df_stats.copy()
    for col in df_stats.columns:
        if col in p_values:
            # 对主统计量使用科学计数法（保留4位小数），p值使用普通格式
            formatted_df[col] = [f"{format(df_stats[col].iloc[i], '.4f')} {p_values[col][i]}" 
                               for i in range(len(df_stats))]
        else:
            # 使用自定义格式化函数确保科学计数法中的数字部分保留4位小数
            def format_scientific(x):
                # 将数字转换为科学计数法字符串
                s = f"{x:e}"
                # 分割科学计数法的数字部分和指数部分
                base, exponent = s.split('e')
                # 确保数字部分保留4位小数
                base = f"{float(base):.4f}"
                # 重新组合
                return f"{base}e{exponent}"
            
            formatted_df[col] = [format_scientific(x) for x in df_stats[col]]
    
    print("\n基本统计量表格:")
    print(formatted_df.to_string())
    
    # 保存到文本文件
    with open('stats_table.txt', 'w', encoding='utf-8') as f:
        f.write("基本统计量表格:\n")
        f.write(formatted_df.to_string())
    
    # 保存到CSV文件
    formatted_df.to_csv('stats_table.csv', encoding='utf-8-sig')  # 使用utf-8-sig编码以支持中文Excel打开
    print("\n统计结果已保存到 stats_table.csv 文件中")

def main():
    """主函数"""
    # 创建plots文件夹
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 加载数据
    print("正在加载数据...")
    df = load_data()
    
    # 计算基本统计量表格
    calculate_basic_stats(df)
    
    # 绘制各类图形
    print("正在生成可视化图表...")
    plot_time_series(df)
    plot_return_series(df)
    plot_distribution(df)
    plot_acf_pacf(df)
    plot_volatility_clustering(df)
    plot_rolling_statistics(df)
    
    # 进行统计检验
    print("\n正在进行统计检验...")
    print("="*50)
    
    # 1. 描述性统计
    calculate_descriptive_stats(df['log_return_usd_jpy'])
    print("="*50)
    
    # 2. 平稳性检验
    perform_stationarity_tests(df['log_return_usd_jpy'])
    print("="*50)
    
    # 3. 正态性检验
    perform_normality_tests(df['log_return_usd_jpy'])
    print("="*50)
    
    # 4. ARCH效应检验
    check_arch_effect(df['log_return_usd_jpy'])
    print("="*50)
    
    # 5. ARCH LM检验
    perform_arch_lm_test(df['log_return_usd_jpy'])
    print("="*50)
    
    # 6. 杠杆效应检验
    check_leverage_effect(df['log_return_usd_jpy'])
    print("="*50)
    
    # 7. 序列相关性检验
    check_serial_correlation(df['log_return_usd_jpy'])
    print("="*50)
    
    print("\n分析完成！所有图表已保存在plots文件夹中。")

if __name__ == "__main__":
    import os
    main() 