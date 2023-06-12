#python3 utils/generate_text.py experiments/roberta_pretrain_sc_cos_adamw/step_100000.ckpt --ema --count 2048
#python3 utils/generate_text.py experiments/roberta_pretrain_sc_cos_adamw/step_200000.ckpt --ema --count 2048
#python3 utils/generate_text.py experiments/roberta_pretrain_sc_cos_adamw/step_300000.ckpt --ema --count 2048
python3 utils/generate_text.py experiments/roberta_pretrain_sc_cos_adamw/step_400000.ckpt --ema --count 2048
#python3 utils/calculate_bloom_loss.py generated_texts/ema_sc_adamw_cosine_sd/step_200000.json
#python3 utils/validate.py experiments/qqp_pretrained/epoch_45.ckpt --ema
#python3 utils/validate.py experiments/sst2_pretrained/epoch_31.ckpt --ema
#python3 utils/validate.py experiments/second_zero_voc2/step_20000.ckpt --ema
