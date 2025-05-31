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

# Import project modules
from data.rxr_dataset import RxRDataset, create_dataloader
from models.clip_adapter import CLIPAdapter
from models.attention_model import MultiHeadAttentionLocalization
from models.heatmap_generator import HeatmapGenerator
from utils.visualization import visualize_heatmap, save_visualization

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test RxR Heatmap Generator on Test Set')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--data_root', type=str, default=None, help='Root directory of RxR dataset')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='CLIP model variant')
    parser.add_argument('--save_visualizations', action='store_true', help='Save visualization of heatmaps')
    parser.add_argument('--save_heatmaps', action='store_true', help='Save raw heatmap data')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(output_dir):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def test(model, dataloader, device, output_dir, save_visualizations=False, save_heatmaps=False):
    """
    Test the model on the test set.
    
    Args:
        model: Model to test
        dataloader: Test dataloader
        device: Device to run testing on
        output_dir: Output directory
        save_visualizations: Whether to save visualizations
        save_heatmaps: Whether to save raw heatmap data
        
    Returns:
        dict: Test results
    """
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Testing')):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            outputs = model(batch)
            pred_heatmaps = outputs['heatmap']
            
            # Process each sample in the batch
            for i in range(len(pred_heatmaps)):
                if batch_idx * dataloader.batch_size + i < len(dataloader.dataset):
                    sample_idx = batch_idx * dataloader.batch_size + i
                    
                    # Extract sample information
                    sample_info = {
                        'sample_idx': sample_idx,
                        'instruction_id': batch['instruction_id'][i].item() if 'instruction_id' in batch else None,
                        'instruction': batch['instruction'][i] if 'instruction' in batch else None,
                        'scan_id': batch['scan_id'][i] if 'scan_id' in batch else None,
                        'path': batch['path'][i] if 'path' in batch else None
                    }
                    
                    # Save heatmap visualization
                    if save_visualizations:
                        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
                        
                        heatmap = pred_heatmaps[i].detach().cpu().numpy()
                        
                        visualization = visualize_heatmap(
                            heatmap,
                            None,  # No ground truth for test set
                            title=f"Sample {sample_idx}: {sample_info['instruction'][:50]}..."
                        )
                        
                        save_path = os.path.join(output_dir, 'visualizations', f'sample_{sample_idx}.png')
                        save_visualization(visualization, save_path)
                        
                        sample_info['visualization_path'] = save_path
                    
                    # Save raw heatmap data
                    if save_heatmaps:
                        os.makedirs(os.path.join(output_dir, 'heatmaps'), exist_ok=True)
                        
                        heatmap = pred_heatmaps[i].detach().cpu().numpy()
                        
                        heatmap_path = os.path.join(output_dir, 'heatmaps', f'sample_{sample_idx}.npy')
                        np.save(heatmap_path, heatmap)
                        
                        sample_info['heatmap_path'] = heatmap_path
                    
                    # Add component heatmaps if available
                    for component_name in ['attention_heatmap', 'pose_trace_heatmap', 'ground_truth_heatmap']:
                        if component_name in outputs and outputs[component_name] is not None:
                            component = outputs[component_name][i].detach().cpu().numpy() if isinstance(outputs[component_name], torch.Tensor) else None
                            
                            if component is not None and save_heatmaps:
                                component_path = os.path.join(output_dir, 'heatmaps', f'sample_{sample_idx}_{component_name}.npy')
                                np.save(component_path, component)
                                sample_info[f'{component_name}_path'] = component_path
                    
                    results.append(sample_info)
    
    # Save results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main(args):
    """Main testing function."""
    # Set up
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_root:
        config['data']['root'] = args.data_root
    
    # Set up output directory and logging
    test_output_dir = os.path.join(args.output_dir, f'test_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(test_output_dir, exist_ok=True)
    logger = setup_logging(test_output_dir)
    
    logger.info(f"Starting testing with config: {config}")
    logger.info(f"Arguments: {args}")
    
    # Create dataloader for test set
    test_dataloader = create_dataloader(
        data_root=config['data']['root'],
        split='test_standard',  # Use the standard test split
        batch_size=args.batch_size,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    logger.info(f"Created test dataloader with {len(test_dataloader)} batches")
    
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
    
    # Test
    logger.info("Starting testing on test set...")
    results = test(
        model=model,
        dataloader=test_dataloader,
        device=args.device,
        output_dir=test_output_dir,
        save_visualizations=args.save_visualizations,
        save_heatmaps=args.save_heatmaps
    )
    
    # Log results summary
    logger.info("Testing completed!")
    logger.info(f"Processed {len(results)} test samples")
    logger.info(f"Results saved to {test_output_dir}")
    
    # Generate summary report
    summary = {
        'num_samples': len(results),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'checkpoint': args.checkpoint,
        'config': config,
        'output_dir': test_output_dir
    }
    
    with open(os.path.join(test_output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info("Summary report generated")

if __name__ == "__main__":
    args = parse_args()
    main(args)
