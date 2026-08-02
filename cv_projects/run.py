import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from config import Config
from dataset import RetailProductDataset
from model import DINOv3MetricModel
from trainer import MetricTrainer
from inference import ProductRetrievalSystem

def main():
    config = Config()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    train_dataset = RetailProductDataset(
        root_dir=config.data_root,
        split='train',
        image_size=config.image_size,
        is_train=True
    )
    
    test_dataset = RetailProductDataset(
        root_dir=config.data_root,
        split='test',
        image_size=config.image_size,
        is_train=False
    )

    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                              shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                             shuffle=False, num_workers=config.num_workers)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    model = DINOv3MetricModel(
        backbone_name=config.backbone_name,
        embedding_size=config.embedding_size,
        freeze_backbone=config.freeze_backbone
    )
    print(f"Модель загружена: {config.backbone_name}")
    trainer = MetricTrainer(model, config, train_loader, val_loader)
    trainer.train()

    print("\n" + "="*50)
    print("Загрузка лучшей модели и оценка на тестовом наборе...")
    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        checkpoints = sorted([f for f in os.listdir(config.checkpoint_dir) if f.endswith('.pth')])
        if checkpoints:
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoints[-1])
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()
    
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(config.device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(batch['label'])
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    from evaluate import compute_fnrm_at_fmr
    fnrm = compute_fnrm_at_fmr(all_embeddings, all_labels, fmr_target=0.0001)
    print(f"✅ Test FNMR@FMR=0.0001: {fnrm:.5f}")

    print("\nСтроим векторную базу для продакшена...")
    retrieval_system = ProductRetrievalSystem(
        model, config, all_embeddings, all_labels
    )
    print("Векторная БД готова!")

if __name__ == "__main__":
    main()