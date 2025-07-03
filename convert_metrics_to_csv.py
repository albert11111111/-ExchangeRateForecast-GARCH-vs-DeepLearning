import os
import numpy as np
import pandas as pd
import glob

def convert_metrics_to_csv():
    # 获取所有metrics.npy文件
    metrics_files = glob.glob('results/**/metrics.npy', recursive=True)
    
    # 定义指标名称
    metric_names = ['MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE']
    
    # 存储所有结果
    all_results = []
    
    # 遍历每个metrics文件
    for metrics_file in metrics_files:
        try:
            # 从文件路径中提取模型信息
            dir_name = os.path.dirname(metrics_file)
            model_info = os.path.basename(dir_name)
            
            # 解析模型信息
            parts = model_info.split('_')
            pred_len = next((p for i, p in enumerate(parts) if i > 0 and parts[i-1].isdigit()), None)
            model_name = next((p for p in parts if p in ['TimeMixer', 'Autoformer', 'iTransformer', 'TimesNet']), 'Unknown')
            
            # 加载.npy文件
            metrics = np.load(metrics_file)
            
            # 创建结果字典
            result = {
                'Model': model_name,
                'Prediction_Length': pred_len,
                'Full_Name': model_info
            }
            
            # 添加指标
            for i, metric in enumerate(metric_names):
                if i < len(metrics):
                    result[metric] = metrics[i]
            
            all_results.append(result)
            
            # 同时也保存单独的CSV
            df_single = pd.DataFrame([{metric_names[i]: metrics[i] for i in range(len(metrics))}])
            csv_file = metrics_file.replace('.npy', '.csv')
            df_single.to_csv(csv_file, index=False)
            print(f'已转换单个文件: {metrics_file} -> {csv_file}')
            
        except Exception as e:
            print(f'处理文件 {metrics_file} 时出错: {str(e)}')
    
    # 创建汇总DataFrame并保存
    if all_results:
        df_all = pd.DataFrame(all_results)
        # 按模型名称和预测长度排序
        df_all = df_all.sort_values(['Model', 'Prediction_Length'])
        summary_file = 'results/all_metrics_summary.csv'
        df_all.to_csv(summary_file, index=False)
        print(f'\n已创建汇总文件: {summary_file}')
        print('\n汇总数据预览:')
        print(df_all.to_string())
        
        # 检查是否所有文件都有相同数量的指标
        metrics_counts = df_all[metric_names].notna().sum(axis=1)
        if not (metrics_counts == len(metric_names)).all():
            print('\n警告：某些文件的指标数量不一致！')
            print('每个文件的指标数量:')
            print(metrics_counts.value_counts())

if __name__ == '__main__':
    convert_metrics_to_csv() 