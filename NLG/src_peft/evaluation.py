import json
import evaluate # 当前不支持 CIDEr, NIST
import argparse
from transformers import GPT2Tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--hypothesis_file", type=str, default="./data/e2e/peft_generated.jsonl", help="Path to the generated data file (JSONL format).")
    parser.add_argument("--reference_file", type=str, default="./data/e2e/test_formatted_400.jsonl", help="Path to the reference data file (JSONL format).")
    args = parser.parse_args()

    hypothesis_file = args.hypothesis_file
    reference_file = args.reference_file

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

    with open(hypothesis_file) as f:
        hypothesis = [json.loads(line)["text"] for line in f]

    with open(reference_file) as f:
        # 这里要保证 decode 的输入是 id list，不是字符串
        # reference = [tokenizer.decode(json.loads(line)["completion"], skip_special_tokens=True) for line in f]
        reference = [json.loads(line)["completion"] for line in f]

    # ➜ 必须保证 hypothesis: List[str]
    # ➜ 必须保证 references: List[List[str]] (外层样本，内层多个参考)
    # 所以要把 reference 变成 [[r], [r], ...]

    # BLEU
    bleu = evaluate.load("bleu")
    bleu_result = bleu.compute(predictions=hypothesis,
                            references=[[r] for r in reference])
    bleu_score = bleu_result['bleu']

    # METEOR
    meteor = evaluate.load("meteor")
    meteor_result = meteor.compute(predictions=hypothesis,
                                references=reference)  # METEOR 可以是 List[str]
    meteor_score = meteor_result['meteor']

    # TER
    ter = evaluate.load("ter")
    ter_result = ter.compute(predictions=hypothesis,
                            references=[[r] for r in reference])
    ter_score = ter_result['score']

    # ROUGE_L
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=hypothesis,
                                references=reference)
    rougeL_score = rouge_result['rougeL']

    # 统一打印
    print("SCORES:")
    print("==============")
    print(f"BLEU: {bleu_score:.4f}")
    print(f"METEOR: {meteor_score:.4f}")
    print(f"TER: {ter_score:.4f}")
    print(f"ROUGE_L: {rougeL_score:.4f}")
