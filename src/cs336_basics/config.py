from dataclasses import dataclass

from omegaconf import MISSING


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    num_layers: int = 6
    theta: float = 10000.0
    ffn_type: str = "swiglu"
    qk_norm: bool = False


@dataclass(frozen=True)
class BaseOptimizerConfig:
    name: str = MISSING
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 2000


@dataclass(frozen=True)
class AdamWConfig(BaseOptimizerConfig):
    name: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass(frozen=True)
class MuonConfig(BaseOptimizerConfig):
    name: str = "muon"
    muon_learning_rate: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.01
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8


@dataclass(frozen=True)
class DataConfig:
    train_data_path: str = "./results/owt_train.bin"
    val_data_path: str = "./results/owt_valid.bin"
    seed: int = 42


@dataclass(frozen=True)
class TrainerConfig:
    batch_size: int = 64
    num_epochs: int = 1
    max_steps: int = 1_000_000
    device: str = "cuda"
    log_period: int = 10
    val_period: int = 5_000_000
    checkpoint_path: str = "checkpoints"
    resume_from_checkpoint: str | None = None
    wandb_project: str = "cs336-language-model"
    wandb_run_name: str | None = None
    use_torch_compile: bool = True
    num_workers: int = 4


@dataclass(frozen=True)
class ProfilerConfig:
    enabled: bool = False
    wait: int = 5
    warmup: int = 2
    active: int = 3
    repeat: int = 3
    dirpath: str = "tb_logs"


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    optimizer: BaseOptimizerConfig
    data: DataConfig
    trainer: TrainerConfig
    profiler: ProfilerConfig
