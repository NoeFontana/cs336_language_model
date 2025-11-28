import os
import sys

import torch
import torch.nn as nn

# Ensure src is in python path
sys.path.append(os.path.abspath("src"))

from cs336.layer import Embedding, Linear, RMSNorm
from cs336.scripts.train_lm import AdamWConfig, MuonConfig, create_optimizer


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 2D Params (Decay candidates)
        self.embedding = Embedding(10, 10)  # (10, 10)
        self.linear = Linear(10, 10)  # weight (10, 10)
        self.attn_proj = Linear(10, 10)  # weight (10, 10)

        # 1D Params (No Decay candidates)
        self.norm = RMSNorm(10)  # gain (10)
        # Linear bias is 1D (But cs336 Linear has no bias)
        # Scalar param
        self.register_parameter("scalar_param", nn.Parameter(torch.tensor(1.0)))


def test_create_optimizer_adamw():
    model = MockModel()
    config = AdamWConfig(name="adamw", weight_decay=0.1, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)

    optimizer = create_optimizer(model, config)

    assert len(optimizer.param_groups) == 2

    # Group 0: Decay
    decay_group = optimizer.param_groups[0]
    assert decay_group["weight_decay"] == 0.1

    decay_params = decay_group["params"]
    # Should contain: embedding.weight, linear.weights, attn_proj.weights
    assert len(decay_params) == 3
    for p in decay_params:
        assert p.ndim >= 2

    # Group 1: No Decay
    nodecay_group = optimizer.param_groups[1]
    assert nodecay_group["weight_decay"] == 0.0

    nodecay_params = nodecay_group["params"]
    # Should contain: norm.gain, scalar_param
    assert len(nodecay_params) == 2
    for p in nodecay_params:
        assert p.ndim < 2


def test_create_optimizer_muon():
    model = MockModel()
    config = MuonConfig(
        name="muon",
        weight_decay=0.1,
        muon_weight_decay=0.01,
        learning_rate=1e-3,
        muon_learning_rate=0.02,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
        adamw_eps=1e-8,
    )

    optimizer = create_optimizer(model, config)

    # Expected Groups:
    # 0: Muon (all >=2D params) with muon-specific settings
    # 1: AdamW (all <2D params) with adamw-specific settings

    assert len(optimizer.param_groups) == 2

    # Group 0: Muon
    muon_group = optimizer.param_groups[0]
    assert muon_group.get("use_muon") is True
    assert muon_group["weight_decay"] == 0.01  # muon_weight_decay
    assert muon_group["lr"] == 0.02  # muon_learning_rate
    assert len(muon_group["params"]) == 3  # embedding, linear.weights, attn_proj.weights
    for p in muon_group["params"]:
        assert p.ndim >= 2

    # Group 1: AdamW NoDecay (<2D)
    adamw_nodecay_group = optimizer.param_groups[1]
    assert adamw_nodecay_group.get("use_muon") is False
    assert adamw_nodecay_group["weight_decay"] == 0.0
    assert adamw_nodecay_group["lr"] == 1e-3  # base learning_rate
    assert len(adamw_nodecay_group["params"]) == 2  # norm.gain, scalar_param
    for p in adamw_nodecay_group["params"]:
        assert p.ndim < 2
