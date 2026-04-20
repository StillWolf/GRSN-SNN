"""MARL 算法占位包（尚未实现）。

当前仓库只实现了论文 arXiv:2404.15597 的 POMDP 单智能体部分。论文的 MARL 实验
（QMIX on SMAC）留待后续接入。

扩展步骤见 docs/MARL_EXTENSION.md。关键点：
- `grsn.policies.rlifs` 的 GRSN/LIF 神经元是与策略无关的 RNN 单元，可直接作为
  QMIX agent 网络的 RNN 骨干。
- 新增 `QMIX_SNN` 类时放在本目录，不要污染单智能体的 `grsn.algorithms` 命名空间。
"""
raise NotImplementedError(
    "MARL support is not implemented in this revision. "
    "See docs/MARL_EXTENSION.md for the planned integration path."
)
