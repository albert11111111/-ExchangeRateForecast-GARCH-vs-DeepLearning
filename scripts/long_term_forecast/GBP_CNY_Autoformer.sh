#!/bin/bash

# 设置GPU，如有需要请修改
export CUDA_VISIBLE_DEVICES=0

model_name=Autoformer

# 预测未来30天
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --model_id GBP_CNY_96_30 \
  --model $model_name \
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

# 预测未来60天
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --model_id GBP_CNY_96_60 \
  --model $model_name \
  --data gbp_cny \
  --features S \
  --target rate \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 60 \
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

# 预测未来90天
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --model_id GBP_CNY_96_90 \
  --model $model_name \
  --data gbp_cny \
  --features S \
  --target rate \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 90 \
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

# 预测未来180天
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --model_id GBP_CNY_96_180 \
  --model $model_name \
  --data gbp_cny \
  --features S \
  --target rate \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 180 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 6 \
  --freq d \
  --inverse 