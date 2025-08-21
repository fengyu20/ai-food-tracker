from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml

@dataclass
class EvaluationConfig:
    """Configuration for the batch evaluation script."""
    
    # --- Paths ---
    validation_dir: str
    training_dir: str
    output_dir: str
    dataset_type: str

    # --- Evaluation Parameters ---
    max_images: Optional[int]
    max_workers: int

    # --- Provider and Model Settings ---
    providers: List[str] = field(default_factory=list)
    api_key: Optional[str] = None
    
    # --- Model Generation Parameters ---
    default_model_parameters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'EvaluationConfig':
        """Loads configuration from a YAML file."""
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data) 