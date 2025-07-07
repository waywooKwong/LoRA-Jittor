# train
time python src_jittor/gpt2_ft.py \
    --train_data ./data/dart/train_5k.jsonl \
    --valid_data ./data/dart/valid_600.jsonl \
    --train_batch_size 4 \
    --grad_acc 1 \
    --valid_batch_size 2 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model_zip.bin \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 1000 \
    --eval_interval 1000 \
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models_jittor/GPT2_M/dart \
    --random_seed 110

# inference
time python src_jittor/gpt2_beam.py \
    --data ./data/dart/test_1k.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models_jittor/GPT2_M/dart/model.6250.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models_jittor/GPT2_M/dart \
    --output_file predict.6250.b10p08r4.jsonl

# decode
python src_jittor/gpt2_decode.py \
        --vocab ./vocab \
        --sample_file ./trained_models_jittor/GPT2_M/dart/predict.6250.b10p08r4.jsonl \
        --input_file ./data/dart/test_formatted_1k.jsonl \
        --ref_type dart \
        --ref_num 6 \
        --output_ref_file eval/GenerationEval/data/references_dart \
        --output_pred_file eval/GenerationEval/data/hypothesis_dart \
        --tokenize --lower

# evaluate
cd ./eval/GenerationEval/
python eval.py \
    -R data/references_dart/reference \
    -H data/hypothesis_dart \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..
