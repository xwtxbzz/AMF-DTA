from .AttentionDTA import AttentionDTA
from .CKDTA import CKDTA
from .DeepDTA import DeepDTA
from .DrugMGR import DrugMGR
from .EDTA import EDTA
from .FingerDTA import FingerDTA
from .GraphDTA import GraphDTA
from .MambaTransDTA import MambaTransDTA
from .WideDTA import WideDTA

# The nine baselines the paper reports: the five rows of Table I and the four
# it quotes only as a range in Sec. III-B. Importing any submodule executes
# this file, so every entry here is loaded on every framework import -- an
# unimportable module in this list breaks the whole baseline path, not just
# its own model.
__all__ = [
    "AttentionDTA",
    "CKDTA",
    "DeepDTA",
    "DrugMGR",
    "EDTA",
    "FingerDTA",
    "GraphDTA",
    "MambaTransDTA",
    "WideDTA",
]
