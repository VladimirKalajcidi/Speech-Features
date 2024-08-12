from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_root: str
    local_data_file: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    size: str


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    size: Path
    data_size: int
    max_steps: int
    language: str
    task: str
    metric: str
    learning_rate: float
    max_steps: int
    generation_max_length: int
    push_to_hub: bool