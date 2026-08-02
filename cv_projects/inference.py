import torch
import numpy as np
import faiss
from PIL import Image
from dataset import RetailProductDataset  

class ProductRetrievalSystem:
    def __init__(self, model, config, gallery_embeddings=None, gallery_labels=None):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.processor = model.get_processor()
        self.transform = self._get_transform()

        if gallery_embeddings is not None:
            self.build_index(gallery_embeddings, gallery_labels)
    
    def _get_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalize_mean, 
                               std=self.config.normalize_std)
        ])
    
    def build_index(self, embeddings, labels):
        """Построение FAISS индекса"""
        embeddings = embeddings.numpy() if torch.is_tensor(embeddings) else embeddings
        embeddings = embeddings.astype(np.float32)
        
        faiss.normalize_L2(embeddings)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) 
        self.index.add(embeddings)
        self.gallery_labels = labels
    
    def add_new_products(self, embeddings, labels):
        """Добавление новых товаров без переобучения"""
        embeddings = embeddings.numpy() if torch.is_tensor(embeddings) else embeddings
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.gallery_labels = np.concatenate([self.gallery_labels, labels])
    
    def predict(self, image, top_k=5):
        """
        Поиск ближайших товаров в векторной БД
        
        Returns:
            distances: [top_k] расстояния (чем меньше, тем ближе)
            labels: [top_k] метки найденных товаров
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError("image must be path or PIL.Image")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy().astype(np.float32)
        faiss.normalize_L2(embedding)
        
        distances, indices = self.index.search(embedding, top_k)

        distances = 1 - distances[0]  
        indices = indices[0]
        
        return distances, self.gallery_labels[indices]