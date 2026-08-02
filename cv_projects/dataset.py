import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RetailProductDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=224, is_train=True):
        """
        Args:
            root_dir: путь к папке rp2k_dataset (например, C:/.../rp2k_dataset)
            split: 'train' или 'test'
            image_size: размер входного изображения
            is_train: использовать аугментации или нет
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.is_train = is_train
        
        self.samples = []
        self.class_names = []
        
        split_dir = os.path.join(root_dir, 'all', split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Папка {split_dir} не найдена. Проверьте структуру.")

        categories = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))]
        self.class_names = sorted(categories)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for cat in categories:
            cat_path = os.path.join(split_dir, cat)
            image_paths = glob.glob(os.path.join(cat_path, '*.jpg')) + \
                          glob.glob(os.path.join(cat_path, '*.png')) + \
                          glob.glob(os.path.join(cat_path, '*.jpeg'))
            for img_path in image_paths:
                self.samples.append((img_path, self.class_to_idx[cat]))
        
        print(f"[{split}] Загружено {len(self.samples)} изображений, {len(self.class_names)} категорий")
        
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        if self.is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop((self.image_size, self.image_size), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'label_name': self.class_names[label]
        }