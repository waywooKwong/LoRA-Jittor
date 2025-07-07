# Log of LoRA reproduction by Weihua

原 LoRA 论文的实验分为两大部分：NLG, NLU。其中：
- NLG 使用 torch 实现，没有进一步的封装。可以使用 Jittor 进行重构。
- NLU 使用已经高度封装的 Huggingface transformer 库实现，不具备复现的可行性。

我的复现分三步实现：
1. 按照原论文参数，缩小数据规模（1/5），对齐论文实验结果
2. Jittor 重构 torch 代码
3. 由于显存的限制，缩小 train/valid batch_szie（1/2），进一步缩小数据规模（1/10），对齐相同参数下 torch, Jittor 实验

整体的 Log 结构：
```
/
```

