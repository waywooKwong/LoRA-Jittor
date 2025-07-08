# 缩小为原来的 1/10*20 
head -n 100 ./data/webnlg_challenge_2017/train.jsonl > ./data/webnlg_challenge_2017/train_100.jsonl
head -n 20 ./data/webnlg_challenge_2017/valid.jsonl > ./data/webnlg_challenge_2017/valid_20.jsonl

head -n 10 ./data/webnlg_challenge_2017/test.jsonl > ./data/webnlg_challenge_2017/test_10.jsonl
head -n 10 ./data/webnlg_challenge_2017/test_formatted.jsonl > ./data/webnlg_challenge_2017/test_formatted_10.jsonl