import json
import torch
import argparse
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--model_name", type=str, default="gpt2-medium", help="Pre-trained model name (e.g., gpt2-medium).")
    parser.add_argument("--lora_model_path", type=str, default="./trained_models_peft/GPT2_M/e2e/checkpoint-5000", help="Path to the LoRA model checkpoint.")
    parser.add_argument("--test_file", type=str, default="./data/e2e/test_400.jsonl", help="Path to the test data file (JSONL format).")
    parser.add_argument("--output_file", type=str, default="./data/e2e/peft_generated.jsonl", help="Path to save the generated data file (JSONL format).")
    args = parser.parse_args()

    model_name = args.model_name
    lora_model_path = args.lora_model_path
    test_file = args.test_file  
    output_file = args.output_file

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
            # Debug: 记录输入prompts的长度
            input_len = inputs.input_ids.shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                # Core Core Debug!!! 250707
                min_length=input_len + 10,  # 保证至少生成20个新 token
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,  # 添加温度参数
                pad_token_id=tokenizer.eos_token_id,
                # eos_token_id=tokenizer.eos_token_id,
                # repetition_penalty=1.0,  # 避免重复
            )
            # 通过 input_len 记录输入 prompt 的长度，
            # 然后从 outputs[0] 中截取 input_len 之后的部分，这才是模型真正“生成”的内容。
            # generated_tokens = outputs[0]
            generated_tokens = outputs[0][input_len:]
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            fout.write(json.dumps({"text": text}) + "\n")

    print(f"推理完成，已保存到 {output_file}")
