import os
import torch
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.nn.utils.rnn import pad_sequence

# 加载数据
def load_jsonl(file_path):
    return Dataset.from_json(file_path)

# tokenize
def tokenize_fn(examples):
    # 如果数据已经是token ids
    if isinstance(examples["context"][0], list):
        input_ids = [c + d for c, d in zip(examples["context"], examples["completion"])]
        labels = [[-100]*len(c) + d for c, d in zip(examples["context"], examples["completion"])]
    else:
        # 如果数据是字符串，需要先tokenize
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        tokenizer.pad_token = tokenizer.eos_token
        
        input_ids = []
        labels = []
        for c, d in zip(examples["context"], examples["completion"]):
            c_ids = tokenizer.encode(c, add_special_tokens=False)
            d_ids = tokenizer.encode(d, add_special_tokens=False)
            input_ids.append(c_ids + d_ids)
            labels.append([-100] * len(c_ids) + d_ids)
    
    attention_mask = [[1]*len(ids) for ids in input_ids]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

# Debug: 数据每条长度不一样，default_data_collator 不能自动 pad，导致报错。
# from transformers import default_data_collator
# Debug: 默认会用 tokenizer.pad_token_id 进行 padding。但数据不是字符串，不能直接用。
# from transformers.data.data_collator import DataCollatorForLanguageModeling
# Core Debug: 但对于 GPT-2，pad token ID-tokenizer.eos_token_id（通常是 50256）
def custom_data_collator(features):
    batch = {}
    pad_token_id = 50256  # GPT2的eos_token_id
    
    for k in features[0].keys():
        if k in ["input_ids", "attention_mask"]:
            batch[k] = pad_sequence([torch.tensor(f[k]) for f in features], 
                                  batch_first=True, padding_value=pad_token_id if k == "input_ids" else 0)
        elif k == "labels":
            batch[k] = pad_sequence([torch.tensor(f[k]) for f in features], 
                                  batch_first=True, padding_value=-100)
        else:
            batch[k] = [f[k] for f in features]
    return batch

def main():
    parser = argparse.ArgumentParser(description="Replication LoRA by PEFT")
    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="Pre-trained model name (e.g., gpt2-medium).")
    parser.add_argument("--train_file", type=str, default="./data/e2e/train_4k.jsonl",help="Path to the training data file (JSONL format).")
    parser.add_argument("--valid_file", type=str, default="./data/e2e/valid_400.jsonl",help="Path to the validation data file (JSONL format).")
    parser.add_argument("--output_dir", type=str, default="./trained_models_peft/GPT2_M/e2e", help="Directory to save the trained model checkpoints.")
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA attention dimension (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability.")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size per device during evaluation.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 parameter.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps.")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluation steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps.")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Logging directory.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for training.")

    args = parser.parse_args()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,                # lora_dim
        lora_alpha=args.lora_alpha,      # lora_alpha
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn","c_proj"],
        bias="none",
        inference_mode=False,
        # init_lora_weights='gaussian',
    )

    # 加载模型 + LoRA
    # Debug: 训练数据中已是token_id, 不需要tokenizer,
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token  # GPT2 没有 pad token，需要手动指定\
    # tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading model: {args.model_name}")
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model = get_peft_model(model, lora_config)

    train_dataset = load_jsonl(args.train_file)
    valid_dataset = load_jsonl(args.valid_file)

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    valid_dataset = valid_dataset.map(tokenize_fn, batched=True)

    # 训练配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        adam_beta2=args.adam_beta2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        warmup_steps=args.warmup_steps,
        eval_strategy="steps", 
        # fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # tokenizer=tokenizer,
        data_collator=custom_data_collator,
    )

    # 开始训练
    print(f"INFO: Start training...")
    trainer.train()
    print(f"INFO: Training completed. Save Path: {args.output_dir}")

if __name__ == "__main__":
    main()