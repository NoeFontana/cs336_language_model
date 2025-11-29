import torch
import torch.optim as optim

from cs336_basics.config import AdamWConfig, BaseOptimizerConfig, MuonConfig
from cs336_basics.optim.muon import Muon


def create_optimizer(model: torch.nn.Module, config: BaseOptimizerConfig) -> torch.optim.Optimizer:
    """
    Constructs the optimizer, splitting parameters into groups for weight decay.

    Logic:
    - Parameters with 2+ dimensions (Matrices) -> Weight Decay applied.
    - Parameters with < 2 dimensions (Vectors: Norms/Biases) -> 0.0 Weight Decay.
    - If Muon is selected, 2D parameters use Muon, 1D parameters use internal AdamW.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    decay_params = [p for p in param_dict.values() if p.ndim >= 2]
    nodecay_params = [p for p in param_dict.values() if p.ndim < 2]

    if config.name == "adamw":
        assert isinstance(config, AdamWConfig)
        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

    elif config.name == "muon":
        assert isinstance(config, MuonConfig)
        return Muon(
            params=decay_params,
            lr=config.muon_learning_rate,
            momentum=config.muon_momentum,
            nesterov=config.muon_nesterov,
            ns_steps=config.muon_ns_steps,
            weight_decay=config.muon_weight_decay,
            adamw_params=nodecay_params,
            adamw_lr=config.learning_rate,
            adamw_betas=(config.adamw_beta1, config.adamw_beta2),
            adamw_eps=config.adamw_eps,
            adamw_weight_decay=0.0,
        )

    else:
        raise ValueError(f"Unknown optimizer: {config.name}")
