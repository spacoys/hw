import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

class DINOv3MetricModel(nn.Module):
    def __init__(self, backbone_name, embedding_size=512, freeze_backbone=False):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name)
        
        self.embedding_size = embedding_size
        self.freeze_backbone = freeze_backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            dummy_out = self.backbone(dummy)
            backbone_dim = dummy_out.last_hidden_state.shape[-1] 

        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        self.l2_norm = nn.functional.normalize
    
    def forward(self, images):
        outputs = self.backbone(images)
        cls_token = outputs.last_hidden_state[:, 0, :] 
        
        embeddings = self.projection(cls_token)
        embeddings = self.l2_norm(embeddings, p=2, dim=1)
        return embeddings
    
    def get_processor(self):
        return self.processor
    
    def extract_features(self, image_tensor):
        """Для инференса — возвращает нормализованный эмбеддинг"""
        self.eval()
        with torch.no_grad():
            return self.forward(image_tensor)