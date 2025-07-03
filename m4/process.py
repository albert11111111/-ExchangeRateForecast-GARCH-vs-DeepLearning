import pandas as pd
import numpy as np

# 读取 CSV
df = pd.read_csv("Daily-train.csv")

# 假设 V1 是序列 ID，V2 是值
# 分组后提取每条序列
grouped = df.groupby("V1")["V2"]

# 构造 values 列表（每条序列为一个 ndarray）
series_list = [group.astype(float).values for _, group in grouped]

# 检查数量是否为 10462
assert len(series_list) == 10462, f"你只有 {len(series_list)} 条序列，不足 10462 条"

# 保存为 .npz 文件（使用 allow_pickle 因为长度不一定一致）
np.savez("training.npz", values=np.array(series_list, dtype=float))  # object 类型可变长
