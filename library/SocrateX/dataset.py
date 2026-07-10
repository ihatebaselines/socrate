import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import v2
import cv2

class Makeset(Dataset):
    """
    Standard SOCRATE dataset for prediction and validation.
    If we only want inference (no labels), `labels` can be None.
    If we want complex training transformations, we can pass them via `transform`.
    """
    def __init__(self, images, labels=None, transform=None, tokenizer=None, pad_id=None, bos_id=None, eos_id=None, height=32):
        self.images = images 
        self.labels = labels
        self.tokenizer = tokenizer
        self.height = height
        
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        if transform is None:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __getitem__(self, idx):
        image_data = self.images[idx]
        
        # Handle image loading based on input (can be path or direct crop)
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            # If it's a numpy array (cv2 crop)
            image = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
            
        w, h = image.size
        new_h = self.height
        new_w = max(1, int(w * new_h / h))
        image = v2.Resize((new_h, new_w))(image)
        image = self.transform(image)

        # If we have labels (during training/evaluation)
        if self.labels is not None and self.tokenizer is not None:
            label = self.labels[idx]
            label = self.tokenizer.encode(label).ids
            label = [self.bos_id] + label + [self.eos_id]
            label = torch.tensor(label, dtype=torch.long)
            return image, label[:-1], label[1:]
            
        return image

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        from torch.nn.utils.rnn import pad_sequence
        
        if self.labels is None:
            images = batch
            batch_size = len(images)
            c = images[0].shape[0]
            h = images[0].shape[1]
            max_w = max(img.shape[2] for img in images)
            new_images = images[0].new_zeros((batch_size, c, h, max_w))
            for i, img in enumerate(images):
                w = img.shape[2]
                new_images[i, :, :, :w] = img
            return new_images
        else:
            images, label1, label2 = zip(*batch)
            
            # Use 0 as fallback pad_id if it's not set
            pad_val = self.pad_id if self.pad_id is not None else 0
            
            label1 = pad_sequence(label1, batch_first=True, padding_value=pad_val)
            label2 = pad_sequence(label2, batch_first=True, padding_value=pad_val)

            batch_size = len(images)
            c = images[0].shape[0]
            h = images[0].shape[1]
            max_w = max(img.shape[2] for img in images)

            new_images = images[0].new_zeros((batch_size, c, h, max_w))

            for i, img in enumerate(images):
                w = img.shape[2]
                new_images[i, :, :, :w] = img

            return new_images, label1, label2

class SmartBatchSampler(Sampler):
    """
    Custom Batch Sampler that groups labels by length
    to minimize the padding required within each batch.
    """
    def __init__(self, labels, batch_size):
        self.batch_size = batch_size
        labels_list = list(labels)
        lengths = [len(str(lbl)) for lbl in labels_list]
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [sorted_indices[i:i + batch_size] for i in range(0, len(sorted_indices), batch_size)]
        
    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

def load_dataset(path):
    """
    Automatically loads images and labels from a supported file (.csv, .txt, .json, .yaml).
    Returns (images, labels) as lists.
    """
    import csv
    import json
    
    images = []
    labels = []
    
    ext = str(path).lower().split('.')[-1]
    
    if ext == 'csv':
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV is empty or missing headers.")
                
            img_col = next((col for col in reader.fieldnames if 'img' in col.lower() or 'image' in col.lower() or 'path' in col.lower()), reader.fieldnames[0])
            lbl_col = next((col for col in reader.fieldnames if 'lbl' in col.lower() or 'label' in col.lower() or 'text' in col.lower() or 'word' in col.lower()), reader.fieldnames[1] if len(reader.fieldnames) > 1 else None)
            
            for row in reader:
                images.append(row[img_col])
                if lbl_col:
                    labels.append(row[lbl_col])
                
    elif ext == 'txt':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    images.append(parts[0])
                    labels.append(parts[1])
                    
    elif ext in ['json']:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                img_key = next((k for k in item.keys() if 'img' in k.lower() or 'path' in k.lower()), None)
                lbl_key = next((k for k in item.keys() if 'lbl' in k.lower() or 'text' in k.lower() or 'word' in k.lower()), None)
                if img_key and lbl_key:
                    images.append(item[img_key])
                    labels.append(item[lbl_key])
                    
    elif ext in ['yaml', 'yml']:
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            for item in data:
                img_key = next((k for k in item.keys() if 'img' in k.lower() or 'path' in k.lower()), None)
                lbl_key = next((k for k in item.keys() if 'lbl' in k.lower() or 'text' in k.lower() or 'word' in k.lower()), None)
                if img_key and lbl_key:
                    images.append(item[img_key])
                    labels.append(item[lbl_key])
    else:
        raise ValueError(f"Unsupported format: {ext}")
        
    return images, labels
