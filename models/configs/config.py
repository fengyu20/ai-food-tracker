from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml

@dataclass
class EvaluationConfig:
    
    validation_dir: str
    training_dir: str
    output_dir: str
    dataset_type: str

    max_images: Optional[int]
    max_workers: int

    providers: List[str] = field(default_factory=list)
    api_key: Optional[str] = None
    
    default_model_parameters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'EvaluationConfig':
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

