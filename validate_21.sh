#python3 utils/generate_text.py experiments/roberta_pretrain_sc_cos_adamw/step_100000.ckpt --ema --count 2048
#python3 utils/generate_text.py experiments/roberta_pretrain_sc_cos_adamw/step_200000.ckpt --ema --count 2048
#python3 utils/generate_text.py experiments/roberta_pretrain_sc_cos_adamw/step_300000.ckpt --ema --count 2048
python3 utils/validate.py experiments/qnli_pretrained/20_0.8785.ckpt --ema
#python3 utils/validate.py experiments/cola_pretrained_roberta/320_0.5312.ckpt --ema
#python3 utils/validate.py experiments/cola_pretrained_roberta_t5t/261_0.6125.ckpt --ema
#python3 utils/validate.py experiments/cola_pretrained_t5t/312_0.5908.ckpt --ema
#python3 utils/calculate_bloom_loss.py generated_texts/ema_sc_adamw_cosine_sd/step_200000.json
#python3 utils/validate.py experiments/qqp_pretrained/epoch_45.ckpt --ema
#python3 utils/validate.py experiments/sst2_pretrained/epoch_31.ckpt --ema
#python3 utils/validate.py experiments/second_zero_voc2/step_20000.ckpt --ema
