# LoRA Replication & Jittor Refactor 

> **LoRA: Low-Rank Adaptation of Large Language Models** <br>
*Edward J. Hu\*, Yelong Shen\*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* <br>
Paper: https://arxiv.org/abs/2106.09685 <br>
Github: https://github.com/microsoft/LoRA <br>

## Introduction of LoRA

LoRA 通过秩（rank）分解矩阵来减少可训练参数的数量，同时冻结原始权重。
* 极大地降低了针对特定任务适配的大语言模型的存储需求
* 并能够在部署期间实现高效的任务切换
* 且不会引入推理延迟

<p>
<img src="figure/readme/lora.png" width="250" >
</p>

|   | Method              | # of Trainable Params | E2E (BLEU)   | DART (BLEU)  | WebNLG (BLEU-U/S/A)            |
|---|---------------------|-----------------------|--------------|--------------|--------------------------------|
|   | GPT-2 M (Fine-Tune) | 354.92M               | 68.2         | 46.0         | 30.4/<b>63.2</b>/47.6          |
|   | GPT-2 M (LoRA)      | 0.35M                 |<b>70.4</b>±.1|<b>47.1</b>±.2| <b>46.7</b>±.4/62.1±.2/<b>55.3</b>±.2 |

## Outline

复现的思路分为以下三个步骤：
1. Stage 1复现并基本调试原仓库的 LoRA 实现代码，验证声明的实验性能与整体。
2. Jittor 重构原 torch 框架的代码，对齐实验性能。
3. (Addition) 使用原论文推荐的现代框架 [PEFT](https://huggingface.co/docs/peft/en/index) 复现 LoRA 实验。

## Innovation

1. 使用当前（截止2025年7月）较新的环境配置复现实验。<br>解决兼容遇到的问题，便于后来者在此基础上使用新配置复现。
2. 详细记录复现过程观察到的指标（如loss，GPU负载，运行时间等）。<br>对比同期复现工作，平衡实验效率和配置需求的基础上，对齐原实验的性能差距最小。
3. LoRA已被更新的一些高效开发框架兼容，尝试使用PEFT复现整体实验流程。

## Experiment Record

### Environment

AutoDL 租用云服务器（推荐内蒙B区，3090资源充足）

* PyTorch 2.3.0 + Python 3.12(ubuntu22.04) + CUDA 12.1
* GPU RTX 3090(24GB) * 1
* CPU 14 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
* 内存 60GB
* 硬盘 系统盘:30 GB 数据盘:免费:50GB

### Repository Overview

原仓库给出两个实例 NLG, NLU:
* [examples/NLG/](https://github.com/microsoft/LoRA/tree/main/examples/NLG): LoRA in GPT-2
* [examples/NLU/](https://github.com/microsoft/LoRA/tree/main/examples/NLU): LoRA in RoBERTa and DeBERTa

其中，
* NLG 基于 torch 实现整体实验流程，适用于使用 Jittor 框架进行重构与性能对齐。
* NLU 使用高度封装的 transformer 实现实验流程，复现的可能性较低。

因此，本仓库进行 NLG 的复现。

### Start up

(optional) set up acadamic mirror in AutoDL
```
source /etc/network_turbo
```

Clone repo
```

```

Install dependencies (recommand virtual environment)
```
sudo apt-get update
···
(activate your virtual environment)
···
pip install -r requirement.txt
```
```
```



## Debug

### 1. Jittor 安装

```
sudo apt install libomp-dev
python -m pip install git+https://github.com/Jittor/jittor.git
```
运行 Jiitor 测试代码：
```
python -m jittor.test.test_example
```
`Error: ImportError: /root/miniconda3/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found `<br>
`Debug: Jittor 需要包含 GLIBCXX_3.4.30 符号版本的 C++ 标准库 (libstdc++.so.6)，当前 Conda 环境中提供的版本过旧，不包含这个符号。`
```
conda install -c conda-forge libstdcxx-ng -y
```
正常运行结果：
```
    step 990, loss = 0.0013174716150388122 {'hold_vars': 13, 'lived_vars': 61, 'lived_ops': 55}
    ……
    step 999, loss = 0.0009948192164301872 {'hold_vars': 13, 'lived_vars': 61, 'lived_ops': 55}
    ----------------------------------------------------------------------
    Ran 1 test in 14.363s
    OK
```


## Contact
Feel feel to contact me(weihua.kwong@mail.nankai.edu.cn) when meeting problems during this replication.

## Citation
```BibTeX
@inproceedings{
hu2022lora,
title={Lo{RA}: Low-Rank Adaptation of Large Language Models},
author={Edward J Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=nZeVKeeFYf9}
}
```