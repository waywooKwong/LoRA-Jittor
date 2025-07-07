# train
time python src_peft/train.py \
    --model_name gpt2-medium \
    --train_file ./data/e2e/train_4k.jsonl \
    --valid_file ./data/e2e/valid_400.jsonl \
    --output_dir ./trained_models_peft/GPT2_M/e2e \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --learning_rate 0.0002 \
    --num_train_epochs 10 \
    --adam_beta2 0.999 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 50 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 2

# inference
time python src_peft/inference.py \
    --model_name gpt2-medium \
    --lora_model_path ./trained_models_peft/GPT2_M/e2e/checkpoint-2500 \
    --test_file ./data/e2e/test_400.jsonl \
    --output_file ./data/e2e/peft_generated.jsonl

# evaluation
python src_peft/evaluation.py \
    --hypothesis_file ./data/e2e/peft_generated.jsonl \
    --reference_file ./data/e2e/test_formatted_400.jsonl
