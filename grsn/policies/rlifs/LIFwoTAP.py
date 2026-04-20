"""LIF without TAP：time_step=4 + rate coding（spike 平均）。

其余结构与 `LIF.py` 的 `LIFCell` 完全一致，仅改变包装器参数。
"""
from ._base import RecurrentSpikingWrapper
from .LIF import LIFCell


class LIFNode(RecurrentSpikingWrapper):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(
            cell_cls=LIFCell,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            time_step=4,
            rate_code=True,
        )
