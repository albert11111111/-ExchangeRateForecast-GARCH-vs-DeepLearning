export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./ \
  --data_path 英镑兑人民币_20250324_102930.csv \
  --model_id exchange_rate_14 \
  --model $model_name \
  --data exchange_rate \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 14 \
  --batch_size 16 \
  --d_model 16 \
  --d_ff 16 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --freq 'd' \
  --seasonal_patterns 'None' 