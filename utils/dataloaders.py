import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def get_cifar10_dataloaders(batch_size=64, num_workers=4, data_dir='./data', 
                           samples_per_class=None, test_samples_per_class=None,
                           seed=42):
    """
    Get CIFAR-10 dataloaders with optional data subset.
    
    Args:
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        data_dir (str): Directory to store/load CIFAR-10 data
        samples_per_class (int, optional): Number of samples per class for training
        test_samples_per_class (int, optional): Number of samples per class for testing
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, test_loader, full_test_loader)
    """
    # CIFAR-10 data transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, 
                                   download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, 
                                  download=True, transform=test_transform)
    
    # Create subsets if specified
    if samples_per_class is not None:
        train_dataset = create_balanced_subset(train_dataset, samples_per_class, seed)
    
    if test_samples_per_class is not None:
        test_subset = create_balanced_subset(test_dataset, test_samples_per_class, seed)
        test_loader = DataLoader(test_subset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers)
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    
    # Full test loader (always the complete test set)
    full_test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, full_test_loader


def create_balanced_subset(dataset, samples_per_class, seed=42):
    """
    Create a balanced subset with specified number of samples per class.
    
    Args:
        dataset: PyTorch dataset
        samples_per_class (int): Number of samples per class
        seed (int): Random seed
        
    Returns:
        Subset: Balanced subset of the dataset
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get all targets
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Find indices for each class
    class_indices = {}
    for class_idx in range(len(np.unique(targets))):
        class_indices[class_idx] = np.where(targets == class_idx)[0]
    
    # Sample indices for each class
    selected_indices = []
    for class_idx, indices in class_indices.items():
        if len(indices) >= samples_per_class:
            selected = np.random.choice(indices, samples_per_class, replace=False)
        else:
            selected = indices  # Use all available samples if less than requested
        selected_indices.extend(selected)
    
    return Subset(dataset, selected_indices)


def get_cifar100_dataloaders(batch_size=64, num_workers=4, data_dir='./data'):
    """
    Get CIFAR-100 dataloaders.
    
    Args:
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        data_dir (str): Directory to store/load CIFAR-100 data
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # CIFAR-100 has same transforms as CIFAR-10
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, 
                                    download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, 
                                   download=True, transform=test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader 