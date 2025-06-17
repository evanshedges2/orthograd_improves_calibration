"""
OrthoGrad Training Script

This script trains models using both standard SGD and OrthoGrad optimizers
on CIFAR-10 dataset and compares their performance.

Usage:
    python train.py --config configs/resnet18_cifar10.json
    python train.py --optimizer sgd --epochs 100 --lr 0.01
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import argparse
import json
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from utils.orthograd import OrthoGrad
from utils.dataloaders import get_cifar10_dataloaders
from utils.metrics import evaluate_model


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_resnet18_for_cifar10():
    """Create ResNet18 model adapted for CIFAR-10."""
    model = models.resnet18(weights=None)
    # Modify for CIFAR-10 (32x32 images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for smaller images
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes for CIFAR-10
    return model


def create_wrn28_for_cifar10():
    """Create Wide ResNet-28-10 for CIFAR-10."""
    # Simplified implementation - you can expand this
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, targets in tqdm(dataloader, desc="Training"):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, test_loader, optimizer, criterion, device, 
                num_epochs=100, save_path=None):
    """Train model and track metrics."""
    train_losses = []
    train_accuracies = []
    test_metrics_history = []
    
    best_test_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluation
        test_metrics = evaluate_model(model, test_loader, device, criterion)
        test_metrics_history.append(test_metrics)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_metrics['accuracy']:.2f}%, Test ECE: {test_metrics['ece']:.4f}")
        
        # Save best model
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model with test accuracy: {best_test_acc:.2f}%")
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_metrics': test_metrics_history,
        'best_test_accuracy': best_test_acc
    }


def plot_training_curves(results, optimizer_name, save_dir=None):
    """Plot training curves."""
    epochs = range(1, len(results['train_losses']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(epochs, results['train_losses'], label='Train Loss')
    test_losses = [m['loss'] for m in results['test_metrics']]
    ax1.plot(epochs, test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{optimizer_name} - Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, results['train_accuracies'], label='Train Accuracy')
    test_accs = [m['accuracy'] for m in results['test_metrics']]
    ax2.plot(epochs, test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{optimizer_name} - Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    
    # ECE curve
    test_eces = [m['ece'] for m in results['test_metrics']]
    ax3.plot(epochs, test_eces, label='Test ECE', color='red')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('ECE')
    ax3.set_title(f'{optimizer_name} - Expected Calibration Error')
    ax3.legend()
    ax3.grid(True)
    
    # Entropy curve
    test_entropies = [m['entropy'] for m in results['test_metrics']]
    ax4.plot(epochs, test_entropies, label='Test Entropy', color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Entropy')
    ax4.set_title(f'{optimizer_name} - Prediction Entropy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{optimizer_name.lower()}_training_curves.png'))
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train models with SGD and OrthoGrad')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'wrn28'])
    parser.add_argument('--optimizer', type=str, default='both', 
                       choices=['sgd', 'orthograd', 'both'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--samples_per_class', type=int, default=None, 
                       help='Number of samples per class (default: use full dataset)')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Override with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Set device and seed
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    set_seed(args.seed)
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    
    # Create data loaders
    train_loader, test_loader, full_test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        samples_per_class=args.samples_per_class,
        seed=args.seed
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    if args.model == 'resnet18':
        model = create_resnet18_for_cifar10()
    elif args.model == 'wrn28':
        model = create_wrn28_for_cifar10()
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    optimizers_to_run = []
    if args.optimizer in ['sgd', 'both']:
        optimizers_to_run.append('sgd')
    if args.optimizer in ['orthograd', 'both']:
        optimizers_to_run.append('orthograd')
    
    results = {}
    
    for opt_name in optimizers_to_run:
        print(f"\n{'='*50}")
        print(f"Training with {opt_name.upper()}")
        print(f"{'='*50}")
        
        # Reset model
        if args.model == 'resnet18':
            model = create_resnet18_for_cifar10()
        elif args.model == 'wrn28':
            model = create_wrn28_for_cifar10()
        model = model.to(device)
        
        # Create optimizer
        if opt_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                                momentum=args.momentum, weight_decay=args.weight_decay)
        elif opt_name == 'orthograd':
            optimizer = OrthoGrad(
                model.parameters(),
                base_optimizer_cls=optim.SGD,
                grad_renormalization=True,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
        
        # Train model
        save_path = os.path.join(args.output_dir, 'checkpoints', 
                                f'{args.model}_{opt_name}_seed{args.seed}.pt')
        
        start_time = time.time()
        opt_results = train_model(model, train_loader, test_loader, optimizer, 
                                criterion, device, args.epochs, save_path)
        end_time = time.time()
        
        opt_results['training_time'] = end_time - start_time
        results[opt_name] = opt_results
        
        print(f"\nTraining completed in {opt_results['training_time']:.2f} seconds")
        print(f"Best test accuracy: {opt_results['best_test_accuracy']:.2f}%")
        
        # Plot training curves
        plot_training_curves(opt_results, opt_name.upper(), args.output_dir)
    
    # Compare results if both optimizers were run
    if len(results) > 1:
        print(f"\n{'='*50}")
        print("COMPARISON RESULTS")
        print(f"{'='*50}")
        
        for opt_name, opt_results in results.items():
            print(f"{opt_name.upper()}: Best Test Accuracy = {opt_results['best_test_accuracy']:.2f}%")
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(results['sgd']['test_metrics']) + 1)
        
        # Accuracy comparison
        for opt_name, opt_results in results.items():
            test_accs = [m['accuracy'] for m in opt_results['test_metrics']]
            ax1.plot(epochs, test_accs, label=opt_name.upper())
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Test Accuracy Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # ECE comparison
        for opt_name, opt_results in results.items():
            test_eces = [m['ece'] for m in opt_results['test_metrics']]
            ax2.plot(epochs, test_eces, label=opt_name.upper())
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('ECE')
        ax2.set_title('Expected Calibration Error Comparison')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'comparison.png'))
        plt.show()
    
    # Save results
    results_path = os.path.join(args.output_dir, f'results_seed{args.seed}.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for opt_name, opt_results in results.items():
            json_results[opt_name] = {
                'best_test_accuracy': opt_results['best_test_accuracy'],
                'training_time': opt_results['training_time'],
                'train_losses': opt_results['train_losses'],
                'train_accuracies': opt_results['train_accuracies']
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main() 