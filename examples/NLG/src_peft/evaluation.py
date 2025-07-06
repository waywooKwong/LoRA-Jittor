import json
import evaluate
from transformers import GPT2Tokenizer

# 假设生成好的输出和参考在 test.jsonl
hypothesis_file = "./data/e2e/peft_generated.jsonl"
reference_file = "./data/e2e/valid_400.jsonl"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

with open(hypothesis_file) as f:
    hypothesis = [json.loads(line)["text"] for line in f]

with open(reference_file) as f:
    # 这里要保证 decode 的输入是 id list，不是字符串
    reference = [tokenizer.decode(json.loads(line)["completion"], skip_special_tokens=True) for line in f]

# ➜ 必须保证 hypothesis: List[str]
# ➜ 必须保证 references: List[List[str]] (外层样本，内层多个参考)
# 所以要把 reference 变成 [[r], [r], ...]

# BLEU
bleu = evaluate.load("bleu")
bleu_result = bleu.compute(predictions=hypothesis,
                           references=[[r] for r in reference])
print(f"BLEU: {bleu_result}")

# METEOR
meteor = evaluate.load("meteor")
meteor_result = meteor.compute(predictions=hypothesis,
                               references=reference)  # METEOR 可以是 List[str]
print(f"METEOR: {meteor_result}")

# TER
ter = evaluate.load("ter")
ter_result = ter.compute(predictions=hypothesis,
                         references=[[r] for r in reference])
print(f"TER: {ter_result}")
