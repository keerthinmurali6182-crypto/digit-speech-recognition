import torch
import torch.nn as nn

class DigitSpeechCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitSpeechCNN, self).__init__()
        
        # Layer 1: Feature map extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Layer 2: Deep pattern learning
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )
        
        # Fully Connected Classifier Layer
        # Input size calculation: With 40x50 inputs, maxpool twice reduces sizes to 10x12
        self.fc = nn.Sequential(
            nn.Linear(32 * 10 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Flatten spatial tensors into a 1D feature array
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits