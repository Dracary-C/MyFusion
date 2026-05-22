"""Planned pipeline: Assessment Reasoning hidden prior + TPGDiff training.

This file is intentionally a scaffold. The next implementation step is to locate
TPGDiff's exact degradation-prior tensor contract, then connect
`AssessPriorAdapter` to that location.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from myfusion.modules.assess_prior import AssessPriorAdapter
from myfusion.modules.latent_qa import load_assessment_hidden, select_hidden

HiddenKey = Literal["prefix_hidden", "generated_hidden", "condition_hidden"]


@dataclass
class AssessTPGDConfig:
    hidden_path: Path
    target_prior_dim: int
    hidden_key: HiddenKey = "condition_hidden"
    pool: str = "mean"


def load_prior_from_hidden(config: AssessTPGDConfig, device: str | torch.device = "cpu") -> torch.Tensor:
    """Load one hidden-state pack and project it to a candidate TPGDiff prior.

    This helper is useful for shape probes before wiring the feature into the
    full training loop.
    """

    pack = load_assessment_hidden(config.hidden_path, map_location="cpu")
    hidden = select_hidden(pack, config.hidden_key).to(device)
    adapter = AssessPriorAdapter(output_dim=config.target_prior_dim, pool=config.pool).to(device)
    return adapter(hidden)


def train_assess_tpgd() -> None:
    """Future full training entrypoint.

    Planned steps:
      1. Load TPGDiff train config and dataset.
      2. Load or precompute Assessment Reasoning hidden states for each sample.
      3. Project hidden states with AssessPriorAdapter.
      4. Replace TPGDiff degradation prior at the selected hook.
      5. Train adapter-only first, then optionally unfreeze restoration modules.
    """

    raise NotImplementedError("Full assess_tpgd training loop is not wired yet.")
