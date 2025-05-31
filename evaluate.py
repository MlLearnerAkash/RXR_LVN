import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
import json
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Import project modules
from data.rxr_dataset import RxRDataset, create_dataloader
from models.clip_adapter import CLIPAdapter
from models.attention_model import MultiHeadAttentionLocalization
from models.heatmap_generator import HeatmapGenerator
from utils.visualization import visualize_heatmap, save_visualization

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate RxR Heatmap Generator')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--data_root', type=str, default=None, help='Root directory of RxR dataset')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='CLIP model variant')
    parser.add_argument('--split', type=str, default='val_unseen', help='Dataset split to evaluate on')
    parser.add_argument('--save_visualizations', action='store_true', help='Save visualization of heatmaps')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate (None for all)')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(output_dir):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def calculate_metrics(pred_heatmaps, target_heatmaps):
    """
    Calculate evaluation metrics for heatmaps.
    
    Args:
        pred_heatmaps (np.ndarray): Predicted heatmaps [N, H, W]
        target_heatmaps (np.ndarray): Target heatmaps [N, H, W]
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Mean Squared Error
    mse = np.mean((pred_heatmaps - target_heatmaps) ** 2)
    metrics['mse'] = mse
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred_heatmaps - target_heatmaps))
    metrics['mae'] = mae
    
    # Structural Similarity Index (simplified version)
    # In a real implementation, you would use skimage.metrics.structural_similarity
    ssim_values = []
    for i in range(len(pred_heatmaps)):
        pred = pred_heatmaps[i].flatten()
        target = target_heatmaps[i].flatten()
        
        # Calculate means
        pred_mean = np.mean(pred)
        target_mean = np.mean(target)
        
        # Calculate variances and covariance
        pred_var = np.var(pred)
        target_var = np.var(target)
        covar = np.mean((pred - pred_mean) * (target - target_mean))
        
        # Constants for stability
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2
        
        # Calculate SSIM
        num = (2 * pred_mean * target_mean + C1) * (2 * covar + C2)
        denom = (pred_mean**2 + target_mean**2 + C1) * (pred_var + target_var + C2)
        ssim = num / denom
        
        ssim_values.append(ssim)
    
    metrics['ssim'] = np.mean(ssim_values)
    
    # Average Precision
    ap_values = []
    for i in range(len(pred_heatmaps)):
        pred = pred_heatmaps[i].flatten()
        target = (target_heatmaps[i] > 0.5).flatten().astype(np.int32)
        
        if np.sum(target) > 0:  # Only calculate if there are positive samples
            ap = average_precision_score(target, pred)
            ap_values.append(ap)
    
    if ap_values:
        metrics['ap'] = np.mean(ap_values)
    else:
        metrics['ap'] = 0.0
    
    # IoU at different thresholds
    iou_thresholds = [0.3, 0.5, 0.7]
    for threshold in iou_thresholds:
        iou_values = []
        for i in range(len(pred_heatmaps)):
            pred_binary = (pred_heatmaps[i] > threshold).astype(np.int32)
            target_binary = (target_heatmaps[i] > threshold).astype(np.int32)
            
            intersection = np.sum(pred_binary & target_binary)
            union = np.sum(pred_binary | target_binary)
            
            if union > 0:
                iou = intersection / union
                iou_values.append(iou)
        
        if iou_values:
            metrics[f'iou_{threshold}'] = np.mean(iou_values)
        else:
            metrics[f'iou_{threshold}'] = 0.0
    
    return metrics

def plot_precision_recall_curve(pred_heatmaps, target_heatmaps, output_path):
    """
    Plot precision-recall curve.
    
    Args:
        pred_heatmaps (np.ndarray): Predicted heatmaps [N, H, W]
        target_heatmaps (np.ndarray): Target heatmaps [N, H, W]
        output_path (str): Path to save the plot
    """
    # Flatten all heatmaps
    all_preds = []
    all_targets = []
    
    for i in range(len(pred_heatmaps)):
        pred = pred_heatmaps[i].flatten()
        target = (target_heatmaps[i] > 0.5).flatten().astype(np.int32)
        
        all_preds.extend(pred)
        all_targets.extend(target)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_targets, all_preds)
    ap = average_precision_score(all_targets, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, lw=2, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def evaluate(model, dataloader, device, output_dir, save_visualizations=False, num_samples=None):
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to run evaluation on
        output_dir: Output directory
        save_visualizations: Whether to save visualizations
        num_samples: Number of samples to evaluate (None for all)
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_pred_heatmaps = []
    all_target_heatmaps = []
    all_metadata = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluation')):
            # Check if we've processed enough samples
            if num_samples is not None and batch_idx * dataloader.batch_size >= num_samples:
                break
            
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            outputs = model(batch)
            pred_heatmaps = outputs['heatmap']
            
            # Create target heatmaps (placeholder, same as in training)
            target_heatmaps = torch.zeros_like(pred_heatmaps)
            for i in range(len(target_heatmaps)):
                h, w = target_heatmaps[i].shape
                center_y, center_x = h // 2, w // 2
                for y in range(h):
                    for x in range(w):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        target_heatmaps[i][y, x] = np.exp(-dist**2 / (2 * (w/8)**2))
                
                # Normalize
                if target_heatmaps[i].max() > 0:
                    target_heatmaps[i] = target_heatmaps[i] / target_heatmaps[i].max()
            
            # Convert to numpy for evaluation
            pred_np = pred_heatmaps.detach().cpu().numpy()
            target_np = target_heatmaps.detach().cpu().numpy()
            
            all_pred_heatmaps.append(pred_np)
            all_target_heatmaps.append(target_np)
            
            # Save metadata for each sample
            for i in range(len(pred_heatmaps)):
                if batch_idx * dataloader.batch_size + i < len(dataloader.dataset):
                    sample_idx = batch_idx * dataloader.batch_size + i
                    metadata = {
                        'sample_idx': sample_idx,
                        'instruction_id': batch['instruction_id'][i].item() if 'instruction_id' in batch else None,
                        'instruction': batch['instruction'][i] if 'instruction' in batch else None
                    }
                    all_metadata.append(metadata)
            
            # Save visualizations
            if save_visualizations:
                os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
                
                for i in range(len(pred_heatmaps)):
                    if batch_idx * dataloader.batch_size + i < len(dataloader.dataset):
                        sample_idx = batch_idx * dataloader.batch_size + i
                        
                        visualization = visualize_heatmap(
                            pred_np[i],
                            target_np[i],
                            title=f'Sample {sample_idx}'
                        )
                        
                        save_path = os.path.join(output_dir, 'visualizations', f'sample_{sample_idx}.png')
                        save_visualization(visualization, save_path)
    
    # Concatenate all heatmaps
    all_pred_heatmaps = np.concatenate(all_pred_heatmaps, axis=0)
    all_target_heatmaps = np.concatenate(all_target_heatmaps, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_pred_heatmaps, all_target_heatmaps)
    
    # Plot precision-recall curve
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    plot_precision_recall_curve(
        all_pred_heatmaps,
        all_target_heatmaps,
        os.path.join(output_dir, 'plots', 'precision_recall_curve.png')
    )
    
    # Save metrics and metadata
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(all_metadata, f, indent=4)
    
    return metrics

def main(args):
    """Main evaluation function."""
    # Set up
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_root:
        config['data']['root'] = args.data_root
    
    # Set up output directory and logging
    eval_output_dir = os.path.join(args.output_dir, f'eval_{args.split}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(eval_output_dir, exist_ok=True)
    logger = setup_logging(eval_output_dir)
    
    logger.info(f"Starting evaluation with config: {config}")
    logger.info(f"Arguments: {args}")
    
    # Create dataloader
    dataloader = create_dataloader(
        data_root=config['data']['root'],
        split=args.split,
        batch_size=args.batch_size,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    logger.info(f"Created dataloader for {args.split} split with {len(dataloader)} batches")
    
    # Initialize models
    clip_adapter = CLIPAdapter(
        model_name=args.clip_model,
        device=args.device
    )
    
    attention_model = MultiHeadAttentionLocalization(
        embed_dim=config['model'].get('embed_dim', 512),
        num_heads=config['model'].get('num_heads', 8),
        dropout=config['model'].get('dropout', 0.1)
    )
    
    model = HeatmapGenerator(
        clip_adapter=clip_adapter,
        attention_model=attention_model,
        embed_dim=config['model'].get('embed_dim', 512),
        use_pose_traces=config['model'].get('use_pose_traces', True),
        use_ground_truth_path=config['model'].get('use_ground_truth_path', True)
    )
    
    model = model.to(args.device)
    logger.info(f"Model initialized and moved to {args.device}")
    
    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        logger.error(f"No checkpoint found at {args.checkpoint}")
        return
    
    # Evaluate
    logger.info("Starting evaluation...")
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        device=args.device,
        output_dir=eval_output_dir,
        save_visualizations=args.save_visualizations,
        num_samples=args.num_samples
    )
    
    # Log metrics
    logger.info("Evaluation completed!")
    logger.info("Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Save metrics summary
    with open(os.path.join(eval_output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("Evaluation Metrics:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    logger.info(f"Results saved to {eval_output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
