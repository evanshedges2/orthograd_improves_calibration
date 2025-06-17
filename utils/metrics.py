import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize


def calculate_accuracy(outputs, targets):
    """Calculate classification accuracy."""
    _, predicted = torch.max(outputs, 1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def calculate_top5_accuracy(outputs, targets):
    """Calculate top-5 classification accuracy."""
    _, top5_pred = outputs.topk(5, 1, True, True)
    top5_pred = top5_pred.t()
    correct = top5_pred.eq(targets.view(1, -1).expand_as(top5_pred))
    correct_k = correct[:5].reshape(-1).float().sum(0, keepdim=True)
    total = targets.size(0)
    return 100.0 * correct_k.item() / total


def calculate_ece(logits, targets, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        logits (torch.Tensor): Model logits
        targets (torch.Tensor): Ground truth labels
        n_bins (int): Number of bins for calibration
        
    Returns:
        float: ECE value
    """
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(targets).float()
    
    # Equally spaced bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculate bin metrics
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        bin_size = in_bin.sum().float()
        
        if bin_size > 0:
            bin_accuracy = accuracies[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            ece += (bin_accuracy - bin_confidence).abs() * bin_size / confidences.size(0)
    
    return ece.item()


def calculate_nll(logits, targets):
    """
    Calculate Negative Log Likelihood loss.
    
    Args:
        logits (torch.Tensor): Model logits
        targets (torch.Tensor): Ground truth labels
        
    Returns:
        float: NLL value
    """
    return F.cross_entropy(logits, targets).item()


def calculate_softmax_entropy(logits, aggregate=True):
    """
    Calculate entropy of softmax outputs.
    
    Args:
        logits (torch.Tensor): Model logits
        aggregate (bool): Whether to return mean entropy
        
    Returns:
        float or torch.Tensor: Entropy value(s)
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=1)
    
    if aggregate:
        return entropy.mean().item()
    return entropy


def calculate_max_logit_magnitude(logits):
    """
    Calculate maximum logit magnitude for each sample.
    
    Args:
        logits (torch.Tensor): Model logits
        
    Returns:
        float: Mean maximum logit magnitude
    """
    max_logits, _ = torch.max(logits.abs(), dim=1)
    return max_logits.mean().item()


def calculate_max_softmax_probability(logits):
    """
    Calculate maximum softmax probability for each sample.
    
    Args:
        logits (torch.Tensor): Model logits
        
    Returns:
        float: Mean maximum softmax probability
    """
    probs = F.softmax(logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    return max_probs.mean().item()


def calculate_logit_variance(logits):
    """
    Calculate variance of logits across classes.
    
    Args:
        logits (torch.Tensor): Model logits
        
    Returns:
        float: Mean logit variance
    """
    logit_var = torch.var(logits, dim=1, unbiased=False)
    return logit_var.mean().item()


def calculate_brier_score(logits, targets):
    """
    Calculate Brier Score.
    
    Args:
        logits (torch.Tensor): Model logits
        targets (torch.Tensor): Ground truth labels
        
    Returns:
        float: Brier score
    """
    probs = F.softmax(logits, dim=1)
    num_classes = probs.size(1)
    
    # Convert targets to one-hot
    targets_one_hot = F.one_hot(targets, num_classes).float()
    
    # Calculate Brier score
    brier_score = torch.mean(torch.sum((probs - targets_one_hot) ** 2, dim=1))
    return brier_score.item()


def calculate_confidence_accuracy_correlation(logits, targets):
    """
    Calculate correlation between confidence and accuracy.
    
    Args:
        logits (torch.Tensor): Model logits
        targets (torch.Tensor): Ground truth labels
        
    Returns:
        float: Pearson correlation coefficient
    """
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(targets).float()
    
    # Convert to numpy for correlation calculation
    confidences_np = confidences.cpu().numpy()
    accuracies_np = accuracies.cpu().numpy()
    
    # Calculate Pearson correlation
    correlation = np.corrcoef(confidences_np, accuracies_np)[0, 1]
    
    # Handle NaN case (when all values are the same)
    if np.isnan(correlation):
        return 0.0
    
    return correlation


class TemperatureScaledModel(torch.nn.Module):
    """
    A wrapper for temperature scaling a model.
    Temperature is learned on the validation set.
    """
    def __init__(self, model):
        super(TemperatureScaledModel, self).__init__()
        self.model = model
        # Initialize temperature to 1.0
        self.temperature = torch.nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        logits = self.model(x)
        # Apply temperature scaling
        return logits / self.temperature
    
    def set_temperature(self, temp):
        """Set temperature to a specific value"""
        self.temperature.data = torch.tensor([temp])


def get_logits_and_labels(model, dataloader, device):
    """Get all logits and labels from a data loader"""
    model.eval()
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            
            all_logits.append(logits)
            all_labels.append(targets)
    
    return torch.cat(all_logits), torch.cat(all_labels)


def find_optimal_temperature(model, val_dataloader, device):
    """
    Find optimal temperature on validation set using NLL loss.
    
    Args:
        model: PyTorch model
        val_dataloader: Validation DataLoader
        device: Device to run on
        
    Returns:
        float: Optimal temperature value
    """
    # Get validation logits and labels
    logits, labels = get_logits_and_labels(model, val_dataloader, device)
    
    # Move to CPU for optimization
    logits_cpu = logits.cpu()
    labels_cpu = labels.cpu()
    
    # Define NLL loss function for temperature
    def nll_loss(temperature):
        # Apply temperature scaling
        scaled_logits = logits_cpu / temperature
        # Calculate loss
        loss = F.cross_entropy(scaled_logits, labels_cpu)
        return loss.item()
    
    # Optimize temperature
    result = minimize(nll_loss, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
    
    optimal_temp = result.x[0]
    
    return optimal_temp


def evaluate_with_temperature_scaling(model, dataloader, device, temperature=1.0):
    """
    Evaluate model with temperature scaling.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        temperature: Temperature value for scaling
        
    Returns:
        dict: Dictionary of evaluation metrics with temperature scaling
    """
    # Get predictions
    logits, labels = get_logits_and_labels(model, dataloader, device)
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    
    # Calculate metrics using existing functions
    metrics = {
        'top1_accuracy': calculate_accuracy(scaled_logits, labels),
        'top5_accuracy': calculate_top5_accuracy(scaled_logits, labels),
        'cross_entropy_loss': F.cross_entropy(scaled_logits, labels).item(),
        'ece': calculate_ece(scaled_logits, labels),
        'brier_score': calculate_brier_score(scaled_logits, labels),
        'predictive_entropy': calculate_softmax_entropy(scaled_logits),
        'max_softmax_prob': calculate_max_softmax_probability(scaled_logits),
        'max_logit_magnitude': calculate_max_logit_magnitude(scaled_logits),
        'logit_variance': calculate_logit_variance(scaled_logits),
        'confidence_accuracy_correlation': calculate_confidence_accuracy_correlation(scaled_logits, labels),
        'temperature': temperature
    }
    
    return metrics


def evaluate_model(model, dataloader, device, criterion=None, val_dataloader=None, 
                   include_temperature_scaling=False):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        criterion: Loss criterion (optional)
        val_dataloader: Validation DataLoader for temperature scaling (optional)
        include_temperature_scaling: Whether to include temperature scaling analysis
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_samples = 0
    
    # Collect all predictions and targets
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            
            # Calculate loss
            loss = criterion(logits, targets)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            # Store predictions
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate base metrics
    metrics = {
        'top1_accuracy': calculate_accuracy(all_logits, all_targets),
        'top5_accuracy': calculate_top5_accuracy(all_logits, all_targets),
        'cross_entropy_loss': total_loss / total_samples,
        'ece': calculate_ece(all_logits, all_targets),
        'brier_score': calculate_brier_score(all_logits, all_targets),
        'predictive_entropy': calculate_softmax_entropy(all_logits),
        'max_softmax_prob': calculate_max_softmax_probability(all_logits),
        'max_logit_magnitude': calculate_max_logit_magnitude(all_logits),
        'logit_variance': calculate_logit_variance(all_logits),
        'confidence_accuracy_correlation': calculate_confidence_accuracy_correlation(all_logits, all_targets),
        # Legacy names for backward compatibility
        'accuracy': calculate_accuracy(all_logits, all_targets),
        'loss': total_loss / total_samples,
        'nll': calculate_nll(all_logits, all_targets),
        'entropy': calculate_softmax_entropy(all_logits),
        'max_logit': calculate_max_logit_magnitude(all_logits)
    }
    
    # Add temperature scaling analysis if requested
    if include_temperature_scaling and val_dataloader is not None:
        try:
            # Find optimal temperature on validation set
            optimal_temp = find_optimal_temperature(model, val_dataloader, device)
            
            # Evaluate on test set with temperature scaling
            temp_scaled_metrics = evaluate_with_temperature_scaling(
                model, dataloader, device, temperature=optimal_temp
            )
            
            # Add temperature scaling results
            metrics['temperature_scaling'] = {
                'optimal_temperature': optimal_temp,
                'before_scaling': {
                    'ece': metrics['ece'],
                    'brier_score': metrics['brier_score'],
                    'cross_entropy_loss': metrics['cross_entropy_loss'],
                    'predictive_entropy': metrics['predictive_entropy']
                },
                'after_scaling': {
                    'top1_accuracy': temp_scaled_metrics['top1_accuracy'],
                    'top5_accuracy': temp_scaled_metrics['top5_accuracy'],
                    'ece': temp_scaled_metrics['ece'],
                    'brier_score': temp_scaled_metrics['brier_score'],
                    'cross_entropy_loss': temp_scaled_metrics['cross_entropy_loss'],
                    'predictive_entropy': temp_scaled_metrics['predictive_entropy'],
                    'max_softmax_prob': temp_scaled_metrics['max_softmax_prob'],
                    'confidence_accuracy_correlation': temp_scaled_metrics['confidence_accuracy_correlation']
                },
                'improvement': {
                    'ece': metrics['ece'] - temp_scaled_metrics['ece'],
                    'brier_score': metrics['brier_score'] - temp_scaled_metrics['brier_score'],
                    'cross_entropy_loss': metrics['cross_entropy_loss'] - temp_scaled_metrics['cross_entropy_loss']
                }
            }
            
        except Exception as e:
            print(f"Warning: Temperature scaling analysis failed: {e}")
            metrics['temperature_scaling'] = None
    
    return metrics


def compute_per_class_accuracy(outputs, targets, num_classes=10):
    """
    Compute per-class accuracy.
    
    Args:
        outputs (torch.Tensor): Model outputs
        targets (torch.Tensor): Ground truth labels
        num_classes (int): Number of classes
        
    Returns:
        numpy.ndarray: Per-class accuracies
    """
    _, predicted = torch.max(outputs, 1)
    per_class_acc = np.zeros(num_classes)
    
    for i in range(num_classes):
        class_mask = targets == i
        if class_mask.sum() > 0:
            per_class_acc[i] = (predicted[class_mask] == targets[class_mask]).float().mean().item()
    
    return per_class_acc 