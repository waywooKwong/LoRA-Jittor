# 缩小为原来的 1/10 
head -n 2000 ./data/webnlg_challenge_2017/train.jsonl > ./data/webnlg_challenge_2017/train_2k.jsonl
head -n 400 ./data/webnlg_challenge_2017/valid.jsonl > ./data/webnlg_challenge_2017/valid_400.jsonl

head -n 200 ./data/webnlg_challenge_2017/test.jsonl > ./data/webnlg_challenge_2017/test_200.jsonl
head -n 200 ./data/webnlg_challenge_2017/test_formatted.jsonl > ./data/webnlg_challenge_2017/test_formatted_200.jsonl