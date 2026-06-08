import torch
import torch.nn as nn
import torch.optim as optim
from data_module import get_dataloaders
from model import MLP
from train import train_model
from utils import plot_learning_curves
import time

def run_baseline(device, train_loader, val_loader, test_loader, epochs=50):
    rates = [0.0, 0.2, 0.5]
    histories = []
    labels = []
    
    for p in rates:
        print(f"\\n--- Training Baseline with Dropout p={p} ---")
        model = MLP(dropout_rate=p).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        
        history = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs, device)
        histories.append(history)
        labels.append(f'Dropout p={p}')
        
    plot_learning_curves(histories, labels, 'Baseline: Dropout on MNIST', 'baseline_dropout.png')
    return histories, labels

def run_wow_angle(device, train_loader, val_loader, test_loader, epochs=50, patience=10):
    weight_decays = [1e-5, 1e-3, 1e-2]
    best_wd = None
    best_val_loss = float('inf')
    best_history = None
    
    print("\\n--- Finding optimal Weight Decay ---")
    for wd in weight_decays:
        print(f"\\nTraining with Weight Decay = {wd}")
        model = MLP(dropout_rate=0.0).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
        
        history = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs, device, patience=patience)
        
        min_val_loss = min(history['val_loss'])
        print(f"Weight Decay {wd} achieved min Val Loss: {min_val_loss:.4f}")
        
        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_wd = wd
            best_history = history
            
    print(f"\\nOptimal Weight Decay found: {best_wd}")
    return best_history, best_wd

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=256)
    
    epochs = 40
    
    baseline_histories, baseline_labels = run_baseline(device, train_loader, val_loader, test_loader, epochs=epochs)
    
    wow_history, best_wd = run_wow_angle(device, train_loader, val_loader, test_loader, epochs=epochs, patience=7)
    
    histories_to_plot = [baseline_histories[0], baseline_histories[2], wow_history]
    labels_to_plot = ['Baseline p=0', 'Baseline p=0.5', f'AdamW WD={best_wd} + ES']
    
    plot_learning_curves(histories_to_plot, labels_to_plot, 'Dropout vs Modern Regularization (2026)', 'wow_angle_comparison.png')
    print("\\nExperiments completed. Results saved in 'results' folder.")

if __name__ == '__main__':
    main()
