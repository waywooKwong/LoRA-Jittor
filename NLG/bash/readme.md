# torch - Jittor Contrast experiment Command

## Notice
我先按照原实验的参数设置， 缩小数据规模为原来的1/5，复现了torch版本的效果，之后进行了Jittor的重构对比实验。

但由于Jittor版本运行原始参数OOM， 所以在对比实验这部分调整了 1) 进一步缩小数据规模，为原来的1/10 2）train/valid batch_size 缩小为原来的 1/2

## 数据处理

## 指令运行