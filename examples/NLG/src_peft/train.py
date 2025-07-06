import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.nn.utils.rnn import pad_sequence

# ✅ 参数
model_name = "gpt2-medium"
train_file = "./data/e2e/train_4k.jsonl"
valid_file = "./data/e2e/valid_400.jsonl"
output_dir = "./trained_models_peft/GPT2_M/e2e"

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,                # lora_dim
    lora_alpha=32,      # lora_alpha
    lora_dropout=0.1
)

# ✅ 加载模型 + LoRA
# Debug: 训练数据中已是token_id, 不需要tokenizer,
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # GPT2 没有 pad token，需要手动指定

model = GPT2LMHeadModel.from_pretrained(model_name)
model = get_peft_model(model, lora_config)

# ✅ 加载数据
def load_jsonl(file_path):
    return Dataset.from_json(file_path)

train_dataset = load_jsonl(train_file)
valid_dataset = load_jsonl(valid_file)

# ✅ tokenize
def tokenize_fn(examples):
    input_ids = [c + d for c, d in zip(examples["context"], examples["completion"])]
    labels = [[-100]*len(c) + d for c, d in zip(examples["context"], examples["completion"])]
    attention_mask = [[1]*len(ids) for ids in input_ids]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

train_dataset = train_dataset.map(tokenize_fn, batched=True)
valid_dataset = valid_dataset.map(tokenize_fn, batched=True)

# ✅ 训练配置
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=0.0002,
    weight_decay=0.01,
    num_train_epochs=5,
    save_steps=1000,
    logging_steps=50,
    warmup_steps=500,
    fp16=True,
    eval_steps=500,
)

# Debug: 数据每条长度不一样，default_data_collator 不能自动 pad，导致报错。
# from transformers import default_data_collator
# Debug: 默认会用 tokenizer.pad_token_id 进行 padding。但数据不是字符串，不能直接用。
# from transformers.data.data_collator import DataCollatorForLanguageModeling
def custom_data_collator(features):
    # features: list of dict
    batch = {}
    for k in features[0].keys():
        if k in ["input_ids", "labels", "attention_mask"]:
            batch[k] = pad_sequence([torch.tensor(f[k]) for f in features], batch_first=True, padding_value=0 if k != "labels" else -100)
        else:
            batch[k] = [f[k] for f in features]
    return batch

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # tokenizer=tokenizer,
    data_collator=custom_data_collator,
)

# ✅ 开始训练
trainer.train()
