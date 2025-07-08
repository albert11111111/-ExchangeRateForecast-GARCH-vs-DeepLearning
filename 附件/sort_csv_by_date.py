import pandas as pd

# --- 用户需要修改的参数 ---
input_csv_file = r'.\dataset\USDCPY.csv'  # 替换为你的CSV文件名
output_csv_file = r'.\dataset\sorted_output_file.csv' # 排序后输出的文件名
date_column_name = 'date' # 替换为你的CSV文件中实际的日期列名

# (可选) 如果日期格式不是标准的 YYYY-MM-DD 或 pandas 难以自动识别，
# 请取消下面这行的注释并提供正确的日期格式字符串。
# 例如:
# date_format_string = '%d/%m/%Y'  # 如果日期是 日/月/年 格式
# date_format_string = '%m-%d-%Y %H:%M:%S' # 如果日期是 月-日-年 时:分:秒 格式
# date_format_string = '%Y%m%d' # 如果日期是 YYYYMMDD 格式
date_format_string = None # 默认让pandas自动推断格式
# --- 参数修改结束 ---

try:
    # 1. 读取CSV文件到pandas DataFrame
    print(f"正在读取文件: {input_csv_file}")
    df = pd.read_csv(input_csv_file)

    # 2. 检查日期列是否存在
    if date_column_name not in df.columns:
        print(f"错误：列名 '{date_column_name}' 在CSV文件中未找到。")
        print(f"文件中的列有: {df.columns.tolist()}")
        exit()

    # 3. 将日期列转换为datetime对象
    #    这样做是为了确保按实际日期值排序，而不是按字符串排序。
    print(f"正在将列 '{date_column_name}' 转换为日期时间格式...")
    if date_format_string:
        df[date_column_name] = pd.to_datetime(df[date_column_name], format=date_format_string, errors='coerce')
    else:
        # errors='coerce' 会将无法解析的日期转换成 NaT (Not a Time)
        df[date_column_name] = pd.to_datetime(df[date_column_name], errors='coerce')

    # 检查是否有日期转换失败 (变成 NaT)
    if df[date_column_name].isnull().any():
        num_failed = df[date_column_name].isnull().sum()
        print(f"警告：有 {num_failed} 个日期值无法被正确解析，它们在排序时可能不会按预期处理或被排在前面/后面。")
        print("请检查这些日期的格式是否与预期的 'date_format_string' (如果提供) 或常规格式一致。")

    # 4. 按日期列对DataFrame进行排序
    #    ascending=True 表示升序，即从前到后
    print(f"正在按列 '{date_column_name}' 排序...")
    df_sorted = df.sort_values(by=date_column_name, ascending=True)

    # 5. 将排序后的DataFrame保存到新的CSV文件
    #    index=False 表示不将DataFrame的索引写入到CSV文件中
    df_sorted.to_csv(output_csv_file, index=False, encoding='utf-8-sig') # utf-8-sig 通常能更好地处理特殊字符并在Excel中正确显示
    print(f"文件已成功排序并保存为: {output_csv_file}")

except FileNotFoundError:
    print(f"错误：输入文件 '{input_csv_file}' 未找到。请检查文件名和路径。")
except Exception as e:
    print(f"处理过程中发生错误：{e}")