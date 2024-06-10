from omegaconf import MISSING
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class FlowConfig:
    fname: str = MISSING
    nsamples: Optional[int] = None
    lr: float = 5e-3
    nsteps: int = 10


@dataclass
class SamplerConfig:
    nmcmc: int = MISSING
    joint: List[str] = MISSING
