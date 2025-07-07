# Replication by PEFT

LoRA 官方仓库提到 LoRA 已被 PEFT 支持，使用这样集成的工具也是现在更便捷的开发方式

所以，这里第三种复现方式，使用 PEFT。

## Notice

分析官方仓库提供的数据，并不是文本形式，而是已经tokenization后的token形式。

这一特点在编码、解码时要额外注意。

p.s. vocab.json 和仓库里提供的是否一致？