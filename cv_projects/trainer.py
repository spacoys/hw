import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_metric_learning import losses, miners, distances, reducers

class MetricTrainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.device)
        
        self.model = self.model.to(self.device)
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.loss_func = self._get_loss(config.loss_type)
        self.miner = self._get_miner(config.miner_type) if config.miner_type else None
        
        self.best_fnrm = float('inf')
        self.best_epoch = 0
    
    def _get_loss(self, loss_type):
        if loss_type == "TripletMarginLoss":
            return losses.TripletMarginLoss(margin=0.2)
        elif loss_type == "MultiSimilarityLoss":
            return losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        elif loss_type == "ArcFaceLoss":
            return losses.ArcFaceLoss(num_classes=len(self.train_loader.dataset.labels), 
                                      embedding_size=self.config.embedding_size)
        else:
            raise ValueError(f"Unknown loss: {loss_type}")
    
    def _get_miner(self, miner_type):
        if miner_type == "MultiSimilarityMiner":
            return miners.MultiSimilarityMiner(epsilon=0.1)
        elif miner_type == "TripletMarginMiner":
            return miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")
        return None
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            embeddings = self.model(images)

            if self.miner:
                hard_pairs = self.miner(embeddings, labels)
                loss = self.loss_func(embeddings, labels, hard_pairs)
            else:
                loss = self.loss_func(embeddings, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                labels = batch['label']
                
                embeddings = self.model(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        from evaluate import compute_fnrm_at_fmr
        fnrm = compute_fnrm_at_fmr(all_embeddings, all_labels, fmr_target=0.0001)
        return fnrm
    
    def train(self):
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            fnrm = self.validate()
            
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, FNMR@FMR=0.0001={fnrm:.5f}")
            
            if fnrm < self.best_fnrm:
                self.best_fnrm = fnrm
                self.best_epoch = epoch
                self._save_checkpoint(epoch, fnrm)
    
    def _save_checkpoint(self, epoch, fnrm):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, f"best_model_epoch{epoch}_fnrm{fnrm:.5f}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fnrm': fnrm,
            'config': self.config,
        }, path)
        print(f"Checkpoint saved: {path}")