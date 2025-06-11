import os
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

def calculate_mse(pred, target):
    """Calculate Mean Squared Error."""
    return np.mean((pred - target) ** 2)

def calculate_mae(pred, target):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(pred - target))

def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union.
    
    Args:
        pred (np.ndarray): Predicted heatmap
        target (np.ndarray): Target heatmap
        threshold (float): Threshold for binary conversion
        
    Returns:
        float: IoU score
    """
    pred_binary = (pred > threshold).astype(np.int32)
    target_binary = (target > threshold).astype(np.int32)
    
    intersection = np.sum(pred_binary & target_binary)
    union = np.sum(pred_binary | target_binary)
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_ap(pred, target, threshold=0.5):
    """
    Calculate Average Precision.
    
    Args:
        pred (np.ndarray): Predicted heatmap
        target (np.ndarray): Target heatmap
        threshold (float): Threshold for binary conversion
        
    Returns:
        float: AP score
    """
    pred_flat = pred.flatten()
    target_flat = (target > threshold).flatten().astype(np.int32)
    
    if np.sum(target_flat) == 0:
        return 0.0
    
    return average_precision_score(target_flat, pred_flat)

def calculate_ssim(pred, target):
    """
    Calculate Structural Similarity Index (simplified version).
    
    Args:
        pred (np.ndarray): Predicted heatmap
        target (np.ndarray): Target heatmap
        
    Returns:
        float: SSIM score
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Calculate means
    pred_mean = np.mean(pred_flat)
    target_mean = np.mean(target_flat)
    
    # Calculate variances and covariance
    pred_var = np.var(pred_flat)
    target_var = np.var(target_flat)
    covar = np.mean((pred_flat - pred_mean) * (target_flat - target_mean))
    
    # Constants for stability
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    
    # Calculate SSIM
    num = (2 * pred_mean * target_mean + C1) * (2 * covar + C2)
    denom = (pred_mean**2 + target_mean**2 + C1) * (pred_var + target_var + C2)
    
    return num / denom

def evaluate_batch(pred_heatmaps, target_heatmaps):
    """
    Evaluate a batch of heatmaps.
    
    Args:
        pred_heatmaps (np.ndarray): Predicted heatmaps [batch_size, H, W]
        target_heatmaps (np.ndarray): Target heatmaps [batch_size, H, W]
        
    Returns:
        dict: Dictionary of metrics
    """
    batch_size = len(pred_heatmaps)
    metrics = {
        'mse': [],
        'mae': [],
        'iou_50': [],
        'ap': [],
        'ssim': []
    }
    
    for i in range(batch_size):
        pred = pred_heatmaps[i]
        target = target_heatmaps[i]
        
        metrics['mse'].append(calculate_mse(pred, target))
        metrics['mae'].append(calculate_mae(pred, target))
        metrics['iou_50'].append(calculate_iou(pred, target, threshold=0.5))
        metrics['ap'].append(calculate_ap(pred, target))
        metrics['ssim'].append(calculate_ssim(pred, target))
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    return avg_metrics
