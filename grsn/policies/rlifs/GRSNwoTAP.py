"""GRSN without TAP：time_step=4 + rate coding。

Cell 结构与 `GRSN.py` 的 `GRSNCell` 完全相同；仅包装器参数不同。
"""
from ._base import RecurrentSpikingWrapper
from .GRSN import GRSNCell


class GRSNNode(RecurrentSpikingWrapper):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(
            cell_cls=GRSNCell,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            time_step=4,
            rate_code=True,
        )
