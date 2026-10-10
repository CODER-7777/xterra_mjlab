"""CPU or CUDA inference from the shipped MoE-CTS training checkpoint.

Recreates the student encoder and actor architecture recorded in the shipped
TorchScript graph. Runtime weights come exclusively from the selected checkpoint
(default: model_90000.pt);
the teacher, critic, and optimizer states are unnecessary for inference.
Input is five 45-value frames, oldest first, matching the stair controller.
"""

from pathlib import Path, PosixPath

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints/svanm2_moe/model_90000.pt"


class MLP(nn.Module):
    def __init__(self, dimensions, activate_last=False):
        super().__init__()
        layers = []
        for index, (inputs, outputs) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(inputs, outputs))
            if index < len(dimensions) - 2 or activate_last:
                layers.append(nn.ELU())
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MLP([225, 512, 256, 2048], activate_last=True)
        self.experts = nn.Conv1d(2048, 256, kernel_size=1, groups=8)

    def forward(self, history):
        features = self.backbone(history).unsqueeze(-1)
        return self.experts(features).squeeze(-1).reshape(-1, 8, 32)


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = Experts()
        self.gating_network = nn.Sequential(MLP([225, 512, 256, 8]), nn.Softmax(dim=-1))

    def forward(self, history):
        weights = self.gating_network(history)
        return (weights.unsqueeze(-1) * self.experts(history)).sum(dim=1)


class StudentMoEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = MoE()

    def forward(self, history):
        return F.normalize(self.moe(history), p=2, dim=-1, eps=1e-12)


class CheckpointPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_moe_encoder = StudentMoEEncoder()
        self.actor = MLP([77, 512, 256, 128, 12])

    def forward(self, history):
        latent = self.student_moe_encoder(history)
        return self.actor(torch.cat([latent, history[:, -45:]], dim=-1))


def load_checkpoint_policy(path: Path, device: str | torch.device = "cpu"):
    """Load inference weights strictly, allowing only known metadata types."""
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Policy device must be cpu or cuda (for example cuda:0)")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a CUDA GPU. "
            "Check nvidia-smi and your CUDA-enabled PyTorch installation, "
            "or explicitly select --device cpu."
        )
    # The training snapshot also contains a Path and NumPy RNG state. Keep
    # weights_only enabled and allowlist these specific metadata types.
    numpy_core = np._core if hasattr(np, "_core") else np.core
    safe_types = [
        PosixPath,
        (numpy_core.multiarray._reconstruct, "numpy._core.multiarray._reconstruct"),
        np.ndarray, np.dtype, type(np.dtype("uint32")),
    ]
    with torch.serialization.safe_globals(safe_types):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    inference_state = {
        key: value for key, value in state.items()
        if key.startswith(("student_moe_encoder.", "actor."))
    }
    policy = CheckpointPolicy()
    policy.load_state_dict(inference_state, strict=True)
    policy.to(device).eval().requires_grad_(False)
    return policy, checkpoint["update"]
