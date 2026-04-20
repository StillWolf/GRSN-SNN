"""脉冲神经元注册表。

仅包含论文 (AAAI'25 GRSN) 中的四种 SNN RNN 单元：
- LIF:        基线 LIF（硬复位, β=0.5, T=1 TAP 对齐）
- LIFwoTAP:   基线 LIF 不对齐变体（T=4 rate coding）
- GRSN:       论文主模型（Eq.17 门控 o_{t-1}, 可学习 β, 软复位, T=1 TAP）
- GRSNwoTAP:  GRSN 无 TAP 变体（T=4 rate coding）
"""
REGISTRY = {}

from .LIF import LIFNode as LIF
from .LIFwoTAP import LIFNode as LIFwoTAP
from .GRSN import GRSNNode as GRSN
from .GRSNwoTAP import GRSNNode as GRSNwoTAP

REGISTRY["LIF"] = LIF
REGISTRY["LIFwoTAP"] = LIFwoTAP
REGISTRY["GRSN"] = GRSN
REGISTRY["GRSNwoTAP"] = GRSNwoTAP
