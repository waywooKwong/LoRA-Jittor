import json
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

# 参数
model_name = "gpt2-medium"
lora_model_path = "./trained_models_peft/GPT2_M/e2e/checkpoint-2500"
test_file = "./data/e2e/test_400.jsonl"
output_file = "./data/e2e/peft_generated.jsonl"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = GPT2LMHeadModel.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 读取测试集
prompts = []
with open(test_file) as f:
    for line in f:
        obj = json.loads(line)
        context_str = tokenizer.decode(obj["context"], skip_special_tokens=True)
        prompts.append(context_str)

# 生成并写文件
with open(output_file, "w") as fout:
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        fout.write(json.dumps({"text": text}) + "\n")

print(f"✅ 推理完成，已保存到 {output_file}")
