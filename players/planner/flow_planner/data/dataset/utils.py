import torch
from dataclasses import dataclass, field
from typing import Any
import json
import numpy as np
from mmengine import fileio
import io

def _empty() -> torch.Tensor:
    return torch.empty(0)

_TENSOR_FIELDS = (
    "ego_current",
    "ego_future",
    "neighbor_past",
    "neighbor_future",
    "racing_lines",
)

@dataclass
class RacingDataSample:
    """
    A single sample for training.
    NOTE: All tensors in a sample are assumed to be on the same device.
    """
    batched: bool

    # input data
    ego_current: torch.Tensor = field(default_factory=_empty)
    ego_future: torch.Tensor = field(default_factory=_empty)
    neighbor_past: torch.Tensor = field(default_factory=_empty)
    neighbor_future: torch.Tensor = field(default_factory=_empty)
    racing_lines: torch.Tensor = field(default_factory=_empty)

    def _iter_tensors(self):
        for name in _TENSOR_FIELDS:
            yield name, getattr(self, name)

    def to(self, target: Any) -> "RacingDataSample":
        """
        Moves all tensors in the data sample to the specified target (device/dtype/etc).
        """
        for name, t in self._iter_tensors():
            setattr(self, name, t.to(target))
        return self

    def repeat(self, num_repeat: int) -> "RacingDataSample":
        """Repeat each sample num_repeat times along batch dimension (A,A,A,B,B,B...)."""
        if num_repeat <= 0:
            raise ValueError("num_repeat must be > 0")

        def as_batched(x):
            return x if self.batched else x.unsqueeze(0)

        def rep(x):
            return as_batched(x).repeat_interleave(num_repeat, dim=0)

        return RacingDataSample(
            batched=True,
            ego_current=rep(self.ego_current),
            ego_future=rep(self.ego_future),
            neighbor_past=rep(self.neighbor_past),
            neighbor_future=rep(self.neighbor_future),
            racing_lines=rep(self.racing_lines),
        )
    
def openjson(path):
    value  = fileio.get_text(path)
    data = json.loads(value)
    return data

def opendata(path):
    
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff)

    return npz_data

def convert_to_model_inputs(data, device):
    tensor_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.dtype == np.bool_:
            tensor_data[k] = torch.tensor(v, dtype=torch.bool).unsqueeze(0).to(device)
        else:
            tensor_data[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)

    return tensor_data