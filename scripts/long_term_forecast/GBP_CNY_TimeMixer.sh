#!/bin/bash

# 设置GPU，如有需要请修改
export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

# 设置公共参数
seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=128
train_epochs=10
patience=10

# 预测未来30天
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --model_id GBP_CNY_${seq_len}_30 \
  --model $model_name \
  --data gbp_cny \
  --features S \
  --target rate \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 30 \
  --e_layers $e_layers \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --freq d \
  --inverse

# 预测未来60天
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --model_id GBP_CNY_${seq_len}_60 \
  --model $model_name \
  --data gbp_cny \
  --features S \
  --target rate \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 60 \
  --e_layers $e_layers \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --freq d \
  --inverse

# 预测未来90天
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --model_id GBP_CNY_${seq_len}_90 \
  --model $model_name \
  --data gbp_cny \
  --features S \
  --target rate \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 90 \
  --e_layers $e_layers \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --freq d \
  --inverse

# 预测未来180天
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --model_id GBP_CNY_${seq_len}_180 \
  --model $model_name \
  --data gbp_cny \
  --features S \
  --target rate \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 180 \
  --e_layers $e_layers \
  --enc_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --freq d \
  --inverse 