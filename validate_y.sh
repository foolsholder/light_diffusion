python3 utils/generate_text_full.py experiments/xsum-baseline-lr5e-5-fixdataset-src256/step_60000.ckpt --ema --N 200
#python3 utils/generate_text.py experiments/roberta_pretrain_sc_cos_adamw/step_200000.ckpt --ema --count 2048
#python3 utils/generate_text.py experiments/roberta_pretrain_sc_cos_adamw/step_300000.ckpt --ema --count 2048
#python3 utils/generate_text.py experiments/roberta_pretrain_sc_cos_adamw/step_400000.ckpt --ema --count 2048
python3 utils/calculate_rouge.py generated_texts/ema_xsum-baseline-lr5e-5-fixdataset-src256/step_60000_200.json
#python3 utils/calculate_roberta_metric.py generated_texts/ema_wiki-t5t-nam-noisy-067-bs512-t2/step_400000_200.json
#python3 utils/calculate_div.py generated_texts/ema_wiki-pretrain-nam-noisy-067-bs512-t2/step_400000_200.json
#python3 utils/validate.py experiments/qqp_pretrained/epoch_45.ckpt --ema
#python3 utils/validate.py experiments/sst2_pretrained/epoch_31.ckpt --ema
#python3 utils/validate.py experiments/second_zero_voc2/step_20000.ckpt --ema
