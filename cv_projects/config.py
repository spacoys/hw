from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    data_root: str = r"C:\Users\Артур\cv_projects\archive\rp2k_dataset"
    
    backbone_name: str = "huyhuung/dinov3-vits16-pretrain-lvd1689m"
    embedding_size: int = 512
    freeze_backbone: bool = False  

    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    device: str = "cuda"  
  
    loss_type: str = "MultiSimilarityLoss" 
    miner_type: str = "MultiSimilarityMiner" 

    image_size: int = 224
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    checkpoint_dir: str = "./checkpoints"