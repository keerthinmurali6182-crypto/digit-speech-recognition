import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class AudioDigitDataset(Dataset):
    def __init__(self, num_samples=600, num_mfcc=40, time_steps=50):
        """
        Generates synthetic speech spectrogram feature matrices representing digits 0-9.
        In production, replace this by loading actual audio WAV files and applying 
        librosa.feature.mfcc().
        """
        np.random.seed(42)
        self.num_samples = num_samples
        self.num_mfcc = num_mfcc
        self.time_steps = time_steps
        
        # Simulate 2D audio spectrogram feature maps (channels=1, MFCCs, time)
        self.data = np.random.randn(num_samples, 1, num_mfcc, time_steps).astype(np.float32)
        # 10 classes corresponding to structural digits 0 through 9
        self.labels = np.random.randint(0, 10, size=num_samples).astype(np.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def get_audio_loaders(batch_size=32):
    dataset = AudioDigitDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader