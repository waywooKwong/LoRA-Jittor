# 缩小 1/10
head -n 5000 ./data/dart/train.jsonl > ./data/dart/train_5k.jsonl
head -n 600 ./data/dart//valid.jsonl > ./data/dart/valid_600.jsonl

head -n 1000 ./data/dart/test.jsonl > ./data/dart/test_1k.jsonl
head -n 1000 ./data/dart/test_formatted.jsonl > ./data/dart/test_formatted_1k.jsonl