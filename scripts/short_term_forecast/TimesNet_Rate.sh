export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

# 1步预测
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 16 \
  --d_ff 16 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --inverse

# # 7步预测
# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Daily' \
#   --model_id rate_7step \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --pred_len 7 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 16 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE' \
#   --inverse

# # 14步预测
# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Daily' \
#   --model_id rate_14step \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --pred_len 14 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 16 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE' \
#   --inverse

# # 30步预测
# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Daily' \
#   --model_id rate_30step \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --pred_len 30 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 16 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE' \
#   --inverse