REGISTRY = {}

from .LIF import RecurrentLIFNode as LIF
from spikingjelly.clock_driven import rnn
from .AdaptiveLIF import RecurrentLIFNode as AdaptiveLIF
from .RecurrentLIF import RecurrentLIFNode as RecurrentLIF
from .LIFwoTAP import RecurrentLIFNode as LIFwoTAP
from .GRSNwoTAP import RecurrentLIFNode as GRSNwoTAP

REGISTRY["LIF"] = LIF
REGISTRY["SpikingGRU"] = rnn.SpikingGRU
REGISTRY["AdaptiveLIF"] = AdaptiveLIF
REGISTRY["RecurrentLIF"] = RecurrentLIF
REGISTRY["LIFwoTAP"] = LIFwoTAP
REGISTRY["GRSNwoTAP"] = GRSNwoTAP
