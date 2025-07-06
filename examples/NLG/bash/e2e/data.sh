# 缩小为原来的 10 倍
head -n 4000 ./data/e2e/train.jsonl > ./data/e2e/train_4k.jsonl
head -n 400 ./data/e2e/valid.jsonl > ./data/e2e/valid_400.jsonl

head -n 400 ./data/e2e/test.jsonl > ./data/e2e/test_400.jsonl
head -n 400 ./data/e2e/test_formatted.jsonl > ./data/e2e/test_formatted_400.jsonl