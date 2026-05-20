import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import get_audio_loaders
from model import DigitSpeechCNN

def run_training():
    # Read pipeline config parameters
    config_path = os.path.join(os.path.dirname(__file__), "../config/audio_config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting digit audio training loop on device: {device}")

    # Initialize data split pipes
    train_loader, val_loader = get_audio_loaders(batch_size=config['training']['batch_size'])

    # Instantiate model, optimizer core, and cross-entropy loss metrics
    model = DigitSpeechCNN(num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = config['training']['epochs']
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            # Forward computation step
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            
            # Backward pass 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(device), labels.to(device)
                outputs = model(waveforms)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {running_loss/len(train_loader):.4f} | Validation Accuracy: {accuracy:.2f}%")

    # Serialize model artifact configuration weights
    torch.save(model.state_dict(), "digit_speech_model.pth")
    print("🎉 Training loop completed successfully! Core weights saved as 'digit_speech_model.pth'")

if __name__ == "__main__":
    run_training()