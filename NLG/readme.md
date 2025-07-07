# NLG: Adapting GPT-2 using LoRA
> Origin Repo: https://github.com/microsoft/LoRA/tree/main/examples/NLG<br>

<p>
<img src="figures/LoRA_GPT2.PNG" width="500" >
</p>

本文件夹详细介绍：
* LoRA 原代码环境配置与实现性能复现
* Jittor 重构代码与性能对齐
* PEFT 搭建的快捷实验流程

## Repository Overview

便于对照论文声明的实验性能，本实验使用 [GPT2-Medium](https://huggingface.co/openai-community/gpt2-medium) 版本.

主要文件目录：

Note: 存放大参数权重文件的 folder 在上传仓库时被 .gitignore 注释掉，所以可能拉取代码时未出现。还是以运行代码后本地目录的情况为准。
 
* Source Code
    * [src/](src) 来自官方仓库的 LoRA 原实现代码，基于 torch 实现，没有使用更高级的封装（相比 NLU，使用 transformer 实现）
    * [src_jittor/](src_jittor) 保持原本的代码结构，使用 Jittor 框架整体重构 torch 代码
    * [src_peft/](src_peft) 使用现代 LoRA 框架 PEFT 搭建的训练、推理、评估实验流程
    * [eval/](eval) 手动下载的模型生成评价函数
* Source
    * [data/](data) 实验涉及的全部数据 train, test, valid 以及复现过程中缩放后的数据，以下划线+数量代表缩放后的全部数量
    * [vocab/](vocab) 对应 GPT-2 的词表。当前主流下载模型会同时下载好词表，但是官方代码只对.bin训练，需要专门放置好词表
* Model Checkpoints
    * [pretrained_checkponits/](pretrained_checkponits)
    * [trained_models/](trained_models) 官方实现训练模型权重存放文件
    * [trained_models_jittor/](trained_models_jittor) Jittor 实现训练模型权重存放文件
    * [trained_models_peft/](trained_models_jittor) PEFT 实现训练模型权重存放文件


## Getting Started

 1. Clone repo, install dependencies in a virtual environment
 ```

 sudo apt-get update
 ···
 (activate your virtual environment)
 ···
 
 pip install -r requirement.txt
 bash download_pretrained_checkpoints.sh
 bash create_datasets.sh
 cd ./eval
 bash download_evalscript.sh
 cd ..
 ```

#### Now we are ready to replicate the results in our paper.

## Replicating Our Result on E2E

1. Train GPT-2 Medium with LoRA (see our paper for hyperparameters for GPT-2 Medium)
```
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --train_batch_size 8 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint ./pretrained_checkpoints/gpt2-medium-pytorch_model.bin \
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
    --lora_dim 4 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --random_seed 110
```

2. Generate outputs from the trained model using beam search:
```
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --batch_size 1 \
    --seq_len 512 \
    --eval_len 64 \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/e2e/model.26289.pt \
    --platform local \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --eos_token_id 628 \
    --work_dir ./trained_models/GPT2_M/e2e \
    --output_file predict.26289.b10p08r4.jsonl
```

3. Decode outputs from step (2)
```
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e/predict.26289.b10p08r4.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt
```

4. Run evaluation on E2E test set

```
python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
```

## Replicating Our Result on WebNLG

1. Follow steps 1 and 2 from E2E pipeline by replacing references to E2E with webnlg (see our paper for hyperparameters)

2. Decode outputs from beam search (step 2 above)
```
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/webnlg/predict.20000.b10p08.jsonl \
    --input_file ./data/webnlg_challenge_2017/test_formatted.jsonl \
    --ref_type webnlg \
    --ref_num 6 \
    --output_ref_file eval/GenerationEval/data/references_webnlg \
    --output_pred_file eval/GenerationEval/data/hypothesis_webnlg \
    --tokenize --lower
```

3. Run evaluation on WebNLG test set
```
cd ./eval/GenerationEval/
python eval.py \
    -R data/references_webnlg/reference \
    -H data/hypothesis_webnlg \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..
```

## Replicating Our Result on DART

1. Follow steps 1 and 2 from E2E pipeline by replacing references to E2E with dart (see our paper for hyperparameters)

2. Decode outputs from beam search (step 2 above)
```
python src/gpt2_decode.py \
        --vocab ./vocab \
        --sample_file ./trained_models/GPT2_M/dart/predict.20000.b10p08.jsonl \
        --input_file ./data/dart/test_formatted.jsonl \
        --ref_type dart \
        --ref_num 6 \
        --output_ref_file eval/GenerationEval/data/references_dart \
        --output_pred_file eval/GenerationEval/data/hypothesis_dart \
        --tokenize --lower
```

3. Run evaluation on Dart test set
```
cd ./eval/GenerationEval/
python eval.py \
    -R data/references_dart/reference \
    -H data/hypothesis_dart \
    -nr 6 \
    -m bleu,meteor,ter 
cd ../..
```

## Citation
```
@misc{hu2021lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Hu, Edward and Shen, Yelong and Wallis, Phil and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Lu and Chen, Weizhu},
    year={2021},
    eprint={2106.09685},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```