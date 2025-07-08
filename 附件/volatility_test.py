import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch.unitroot import ADF
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import het_arch
import seaborn as sns
from scipy import stats

# 设置中文显示和图形风格
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 10)

# 读取数据
df = pd.read_csv(r'F:\系统建模与仿真\new-Timeserires\Time-Series-Library-main\Time-Series-Library-main\dataset\sorted_output_file.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 计算对数收益率
df['returns'] = np.log(df['rate']).diff()
df = df.dropna()

# 计算20日滚动波动率
df['volatility'] = df['returns'].rolling(window=20).std()

# 基本统计描述
print("\n基本统计描述：")
print(df['returns'].describe())

# 正态性检验
_, p_value = stats.normaltest(df['returns'])
print(f"\nD'Agostino-Pearson正态性检验 p值: {p_value:.4f}")

# ARCH效应检验
lm_stat, p_value, f_stat, f_p_value = het_arch(df['returns'].dropna())
print(f"\nARCH效应检验结果：")
print(f"LM统计量: {lm_stat:.4f}")
print(f"LM检验p值: {p_value:.4f}")
print(f"F统计量: {f_stat:.4f}")
print(f"F检验p值: {f_p_value:.4f}")

# 创建第一组图形：基本时间序列分析
plt.figure(figsize=(15, 12))

# 子图1：原始汇率和20日移动平均
plt.subplot(2, 2, 1)
plt.plot(df.index, df['rate'], label='原始汇率')
plt.plot(df.index, df['rate'].rolling(window=20).mean(), label='20日移动平均')
plt.title('汇率时间序列与趋势')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# 子图2：收益率时间序列
plt.subplot(2, 2, 2)
plt.plot(df.index, df['returns'])
plt.title('对数收益率时间序列')
plt.xticks(rotation=45)
plt.grid(True)

# 子图3：收益率分布与正态分布对比
plt.subplot(2, 2, 3)
sns.histplot(data=df['returns'], kde=True, stat='density')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
plt.plot(x, stats.norm.pdf(x, df['returns'].mean(), df['returns'].std()), 
         'r-', label='正态分布')
plt.title('收益率分布与正态分布对比')
plt.legend()
plt.grid(True)

# 子图4：QQ图
plt.subplot(2, 2, 4)
stats.probplot(df['returns'], dist="norm", plot=plt)
plt.title('收益率QQ图')
plt.grid(True)

plt.tight_layout()
plt.savefig('volatility_analysis_1.png', dpi=300, bbox_inches='tight')
plt.close()

# 创建第二组图形：波动性分析
plt.figure(figsize=(15, 12))

# 子图1：20日滚动波动率
plt.subplot(2, 2, 1)
plt.plot(df.index, df['volatility'])
plt.title('20日滚动波动率')
plt.xticks(rotation=45)
plt.grid(True)

# 子图2：收益率的自相关图
plt.subplot(2, 2, 2)
acf_values = acf(df['returns'], nlags=40)
plt.bar(range(len(acf_values)), acf_values)
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=1.96/np.sqrt(len(df)), linestyle='--', color='red')
plt.axhline(y=-1.96/np.sqrt(len(df)), linestyle='--', color='red')
plt.title('收益率的自相关函数(ACF)')
plt.grid(True)

# 子图3：收益率平方的自相关图
plt.subplot(2, 2, 3)
acf_values_squared = acf(df['returns']**2, nlags=40)
plt.bar(range(len(acf_values_squared)), acf_values_squared)
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=1.96/np.sqrt(len(df)), linestyle='--', color='red')
plt.axhline(y=-1.96/np.sqrt(len(df)), linestyle='--', color='red')
plt.title('收益率平方的自相关函数(ACF)')
plt.grid(True)

# 子图4：波动率聚集散点图
plt.subplot(2, 2, 4)
plt.scatter(df['volatility'][:-1], df['volatility'][1:], alpha=0.5)
plt.xlabel('t期波动率')
plt.ylabel('t+1期波动率')
plt.title('波动率聚集散点图')
plt.grid(True)

plt.tight_layout()
plt.savefig('volatility_analysis_2.png', dpi=300, bbox_inches='tight')
plt.close()

# 计算波动率聚集指标
rolling_std = df['returns'].rolling(window=20).std()
volatility_clustering = np.corrcoef(rolling_std[20:], rolling_std[19:-1])[0,1]
print(f"\n波动率聚集系数（20日滚动标准差的自相关）: {volatility_clustering:.4f}")

# 计算超额峰度
kurtosis = stats.kurtosis(df['returns'].dropna())
print(f"\n收益率序列的超额峰度: {kurtosis:.4f}")

# 计算偏度
skewness = stats.skew(df['returns'].dropna())
print(f"\n收益率序列的偏度: {skewness:.4f}") 